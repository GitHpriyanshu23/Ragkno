import importlib
import json
import os
import time
import asyncio
import threading

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.sqlite3'}")
    monkeypatch.setenv("RAGKNO_SESSION_SECRET", "test-secret")
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:5173")

    import src.database as database_module
    database_module.Database._instance = None
    import backend.main as main
    main = importlib.reload(main)

    for uid in ("alice", "bob"):
        main._db.upsert_user({"sub": uid, "email": f"{uid}@example.com", "name": uid.title()})

    def auth(uid):
        csrf = f"csrf-{uid}"
        token = main._sign_payload({
            "sub": uid,
            "email": f"{uid}@example.com",
            "name": uid.title(),
            "csrf": csrf,
            "exp": int(time.time()) + 3600,
        })
        return {"cookies": {main.SESSION_COOKIE_NAME: token}, "headers": {"X-CSRF-Token": csrf, "Origin": "http://localhost:5173"}}

    return main, TestClient(main.app), auth


def test_data_routes_require_authentication(api):
    _, client, _ = api
    assert client.get("/threads").status_code == 401
    assert client.get("/ingest/sources").status_code == 401
    assert client.get("/feedback").status_code == 401
    assert client.post("/query", json={"query": "x", "thread_id": "t", "request_id": "request-1"}).status_code == 401


def test_upload_parsing_does_not_block_session_requests(api, monkeypatch):
    import httpx

    main, _, auth = api
    started = threading.Event()
    release = threading.Event()

    def slow_parse(*_args):
        started.set()
        release.wait(5)
        return []

    monkeypatch.setattr(main, "load_uploaded_file", slow_parse)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            upload = asyncio.create_task(client.post(
                "/ingest/files", files={"files": ("notes.txt", b"notes", "text/plain")}, **auth("alice"),
            ))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                response = await asyncio.wait_for(client.get("/auth/me", **auth("alice")), 2)
                assert response.status_code == 200
                assert response.json()["authenticated"] is True
                assert not upload.done(), "Session requests must finish while parsing is still running"
            finally:
                release.set()
                await upload

    asyncio.run(exercise())


def test_csrf_and_thread_ownership_are_enforced(api):
    main, client, auth = api
    alice = auth("alice")
    bob = auth("bob")
    created = client.post("/threads", json={"title": "Alice thread"}, **alice)
    assert created.status_code == 200
    thread_id = created.json()["thread"]["id"]

    assert client.post("/threads", json={"title": "No token"}, cookies=alice["cookies"]).status_code == 403
    assert client.get(f"/threads/{thread_id}/messages", cookies=bob["cookies"]).status_code == 404
    with pytest.raises(PermissionError):
        main._db.append_message(thread_id, "bob", "user", "foreign write")


def test_feedback_is_private_to_submitter(api):
    _, client, auth = api
    assert client.post("/feedback", json={"rating": "positive", "feedback": "Alice note"}, **auth("alice")).status_code == 200
    assert client.post("/feedback", json={"rating": "negative", "feedback": "Bob note"}, **auth("bob")).status_code == 200
    alice_feedback = client.get("/feedback", cookies=auth("alice")["cookies"]).json()["feedbacks"]
    assert [item["feedback"] for item in alice_feedback] == ["Alice note"]


def test_query_request_is_idempotent(api, monkeypatch):
    main, client, auth = api
    created = client.post("/threads", json={"title": "Query"}, **auth("alice")).json()["thread"]

    class FakeRag:
        calls = 0
        def answer_with_sources(self, **_kwargs):
            self.calls += 1
            return {"answer": "Grounded answer [1]", "sources": [{"index": 1, "source": "guide.txt", "text": "Grounded answer", "score": 1.0}]}
        def summarize_history(self, *_args):
            return ""

    fake = FakeRag()
    monkeypatch.setattr(main, "_get_rag", lambda: fake)
    payload = {"query": "What is grounded?", "thread_id": created["id"], "request_id": "request-idempotent", "top_k": 3}
    first = client.post("/query", json=payload, **auth("alice"))
    second = client.post("/query", json=payload, **auth("alice"))
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["replayed"] is True
    assert fake.calls == 1
    messages = main._db.get_thread_messages(created["id"], "alice")
    assert len(messages) == 2

    conflicting = client.post(
        "/query",
        json={**payload, "query": "A different request with the same ID"},
        **auth("alice"),
    )
    assert conflicting.status_code == 409
    assert fake.calls == 1


@pytest.mark.parametrize("endpoint,partial", [("/query", ""), ("/query/stream", ""), ("/query/stream", "Priyanshu is ")])
def test_failed_query_survives_reload_and_retry(api, monkeypatch, endpoint, partial):
    main, client, auth = api
    thread = client.post("/threads", json={"title": "New Chat"}, **auth("alice")).json()["thread"]

    class FakeRag:
        fail = True

        def answer_with_sources(self, **_kwargs):
            if self.fail:
                raise RuntimeError("Provider returned no answer")
            return {"answer": "Priyanshu is the author.", "sources": []}

        def stream_answer_with_sources(self, **_kwargs):
            def tokens():
                if self.fail:
                    if partial:
                        yield partial
                    raise RuntimeError("Provider returned no answer")
                yield "Priyanshu is the author."
            return [], tokens()

        def summarize_history(self, *_args):
            return ""

    rag = FakeRag()
    monkeypatch.setattr(main, "_get_rag", lambda: rag)
    payload = {"query": "Who is Priyanshu?", "thread_id": thread["id"], "request_id": "failed-then-retried"}
    failed = client.post(endpoint, json=payload, **auth("alice"))
    if endpoint.endswith("stream"):
        assert "event: error" in failed.text
    else:
        assert failed.status_code == 500

    messages_url = f"/threads/{thread['id']}/messages"
    messages = client.get(messages_url, cookies=auth("alice")["cookies"]).json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["text"] == payload["query"]
    assert messages[1]["text"] == (partial.strip() or "The response failed before it completed.")
    assert messages[1]["action"]["interrupted"] is True
    assert client.get(messages_url, cookies=auth("bob")["cookies"]).status_code == 404

    rag.fail = False
    retried = client.post(endpoint, json=payload, **auth("alice"))
    assert retried.status_code == 200
    if endpoint.endswith("stream"):
        assert "event: done" in retried.text
    messages = client.get(messages_url, cookies=auth("alice")["cookies"]).json()["messages"]
    assert len(messages) == 2
    assert messages[1]["text"] == "Priyanshu is the author."
    assert messages[1]["action"] is None


def test_simple_greeting_is_answered_without_loading_rag(api, monkeypatch):
    main, client, auth = api
    created = client.post("/threads", json={"title": "Greeting"}, **auth("alice")).json()["thread"]

    def fail_if_rag_loads():
        raise AssertionError("A simple greeting must not initialize or call the RAG/LLM pipeline")

    monkeypatch.setattr(main, "_get_rag", fail_if_rag_loads)
    payload = {
        "query": "Hi!",
        "thread_id": created["id"],
        "request_id": "request-local-greeting",
    }

    response = client.post("/query", json=payload, **auth("alice"))
    assert response.status_code == 200
    body = response.json()
    assert body["local"] is True
    assert body["sources"] == []
    assert body["answer"].startswith("Hi Alice!")
    assert "provide a little more context" in body["answer"]

    messages = main._db.get_thread_messages(created["id"], "alice")
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_streamed_greeting_is_local_and_skips_rag(api, monkeypatch):
    main, client, auth = api
    created = client.post("/threads", json={"title": "Greeting stream"}, **auth("alice")).json()["thread"]

    monkeypatch.setattr(
        main,
        "_get_rag",
        lambda: (_ for _ in ()).throw(AssertionError("RAG should not load for greetings")),
    )
    response = client.post(
        "/query/stream",
        json={
            "query": "hello there",
            "thread_id": created["id"],
            "request_id": "request-local-greeting-stream",
        },
        **auth("alice"),
    )

    assert response.status_code == 200
    assert 'event: meta' in response.text
    assert 'event: token' in response.text
    assert 'event: done' in response.text
    assert '"local": true' in response.text
    streamed_tokens = []
    for block in response.text.split("\n\n"):
        if not block.startswith("event: token"):
            continue
        data_line = next(line for line in block.splitlines() if line.startswith("data: "))
        streamed_tokens.append(json.loads(data_line.removeprefix("data: "))["token"])
    assert "provide a little more context" in "".join(streamed_tokens)


def test_rrf_uses_stable_chunk_ids_and_scores_are_finite():
    from src.search import RAGSearch

    dense = [{"id": "chunk-a", "index": 0, "distance": 0.2, "metadata": {"text": "A"}}]
    sparse = [{"id": "chunk-b", "index": 0, "distance": None, "bm25_score": 3.0, "metadata": {"text": "B"}}]
    fused = RAGSearch._rrf_fuse(dense, sparse)
    assert {item["id"] for item in fused} == {"chunk-a", "chunk-b"}
    assert all(item.get("distance") is None or item["distance"] != float("inf") for item in fused)


def test_agentrouter_stream_ignores_empty_compatibility_events(monkeypatch):
    from src.search import RAGSearch

    class FakeResponse:
        ok = True
        status_code = 200
        text = ""
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is True
            return iter([
                'data: {"choices":[null]}',
                'data: {"choices":[{"delta":null}]}',
                'data: {"choices":[]}',
                'data: {"choices":[{"delta":{"content":"Grounded "}}]}',
                'data: {"choices":[{"delta":{"content":"answer"}}]}',
                'data: [DONE]',
                'data: {"choices":[{"delta":{"content":" ignored"}}]}',
            ])

    monkeypatch.setattr("src.search.requests.post", lambda *_args, **_kwargs: FakeResponse())
    rag = RAGSearch.__new__(RAGSearch)
    rag.agentrouter_base_url = "https://agentrouter.test/v1"
    rag.agentrouter_api_key = "test-key"
    rag.agentrouter_headers = {"x-app": "test"}
    rag.connect_timeout_seconds = 10
    rag.request_timeout_seconds = 15

    assert "".join(rag._stream_agentrouter_response("question", "test-model")) == "Grounded answer"


@pytest.mark.parametrize("content_type,body,expected", [
    ("application/json", '{"choices":[{"message":{"content":"Answer from JSON"}}]}', "Answer from JSON"),
    ("text/html", '<html>Gateway page</html>', None),
    ("application/json", '{"error":{"message":"quota exhausted"}}', None),
])
def test_agentrouter_handles_non_sse_responses(monkeypatch, content_type, body, expected):
    import requests
    from src.search import RAGSearch

    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = content_type
    response._content = body.encode()
    response._content_consumed = True
    monkeypatch.setattr("src.search.requests.post", lambda *_args, **_kwargs: response)
    rag = RAGSearch.__new__(RAGSearch)
    rag.agentrouter_base_url = "https://agentrouter.test/v1"
    rag.agentrouter_api_key = "test-key"
    rag.connect_timeout_seconds = 10
    rag.request_timeout_seconds = 15
    if expected:
        assert "".join(rag._stream_agentrouter_response("question", "test-model")) == expected
    else:
        with pytest.raises(RuntimeError, match="unexpected response format|quota exhausted"):
            list(rag._stream_agentrouter_response("question", "test-model"))


def test_agentrouter_nonstream_calls_share_the_compatible_parser(monkeypatch):
    from src.search import RAGSearch

    rag = RAGSearch.__new__(RAGSearch)
    rag.provider = "agentrouter"
    rag.llm_model = "primary-model"
    rag.fallback_models = []
    rag.allowed_models = frozenset({"primary-model"})
    monkeypatch.setattr(rag, "_stream_agentrouter_response", lambda *_args: iter(["Summary ", "complete"]))

    response = rag._invoke_with_fallback("summarize this")
    assert response.content == "Summary complete"


@pytest.mark.parametrize("failure", [
    "AgentRouter timed out while waiting for a response",
    "empty",
])
def test_agentrouter_timeout_falls_back_before_any_token(monkeypatch, failure):
    from src.search import RAGSearch

    rag = RAGSearch.__new__(RAGSearch)
    rag.provider = "agentrouter"
    rag.llm_model = "slow-model"
    rag.fallback_models = ["fallback-model"]
    rag.allowed_models = frozenset({"slow-model", "fallback-model"})

    def fake_stream(_prompt, model):
        if model == "slow-model":
            if failure == "empty":
                return
            raise RuntimeError(failure)
        yield "Fallback answer"

    monkeypatch.setattr(rag, "_stream_agentrouter_response", fake_stream)
    monkeypatch.setattr("src.search.time.sleep", lambda *_args: None)
    assert "".join(rag._stream_with_fallback("question")) == "Fallback answer"


def test_agentrouter_transient_failure_retries_before_fallback(monkeypatch):
    from src.search import RAGSearch

    rag = RAGSearch.__new__(RAGSearch)
    rag.provider = "agentrouter"
    rag.llm_model = "primary-model"
    rag.fallback_models = ["fallback-model"]
    rag.allowed_models = frozenset({"primary-model", "fallback-model"})
    attempts = []

    def fake_stream(_prompt, model):
        attempts.append(model)
        if len(attempts) == 1:
            raise RuntimeError("SSL handshake timed out")
        yield "Recovered answer"

    monkeypatch.setattr(rag, "_stream_agentrouter_response", fake_stream)
    monkeypatch.setattr("src.search.time.sleep", lambda *_args: None)

    assert "".join(rag._stream_with_fallback("question")) == "Recovered answer"
    assert attempts == ["primary-model", "primary-model"]


@pytest.mark.parametrize("streaming", [True, False])
def test_gemini_chain_uses_agentrouter_last_and_skips_exhausted_models(monkeypatch, streaming):
    from src.search import RAGSearch
    from langchain_core.messages import AIMessage

    models = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("AGENTROUTER_API_KEY", "test-router-key")
    monkeypatch.setenv("AGENTROUTER_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_AGENTROUTER_FALLBACK_ENABLED", "true")
    monkeypatch.delenv("GOOGLE_LLM_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_LLM_FALLBACK_MODELS", raising=False)
    monkeypatch.setattr("src.search.ChromaVectorStore", lambda **_kwargs: object())
    attempts = []

    class FakeGoogle:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            assert kwargs["max_retries"] == 0
        def invoke(self, _prompt):
            attempts.append(self.model)
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded; retry in 120s")
        def stream(self, prompt):
            self.invoke(prompt)
            yield AIMessage(content="unreachable")

    monkeypatch.setattr("src.search.ChatGoogleGenerativeAI", FakeGoogle)
    rag = RAGSearch()

    def router(_prompt, model):
        attempts.append(model)
        yield "Backup answer"
    monkeypatch.setattr(rag, "_stream_agentrouter_response", router)
    call = lambda: "".join(rag._stream_with_fallback("question")) if streaming else rag._invoke_with_fallback("question").content
    assert call() == "Backup answer"
    assert attempts == [*models, "deepseek-v4-flash"]
    assert call() == "Backup answer"
    assert attempts == [*models, "deepseek-v4-flash", "deepseek-v4-flash"]
    assert rag._model_cooldowns[("google", models[0])] - time.monotonic() > 115


def test_gemini_partial_answer_never_switches_providers(monkeypatch):
    from src.search import RAGSearch
    from langchain_core.messages import AIMessage
    rag = RAGSearch.__new__(RAGSearch)
    rag.provider = "google"
    rag.llm_model = "gemini-3.5-flash-lite"
    rag.fallback_models = ["gemini-3.1-flash-lite"]
    rag.allowed_models = frozenset([rag.llm_model, *rag.fallback_models])
    rag.agentrouter_fallback_enabled = True
    rag.agentrouter_api_key = "test"
    rag.agentrouter_fallback_model = "deepseek-v4-flash"

    class PartialGoogle:
        def stream(self, _prompt):
            yield AIMessage(content="Partial answer")
            raise RuntimeError("429 quota exhausted")
    monkeypatch.setattr(rag, "_build_llm", lambda _model: PartialGoogle())
    monkeypatch.setattr(rag, "_stream_agentrouter_response", lambda *_args: pytest.fail("Must not append a different provider's answer"))
    stream = rag._stream_with_fallback("question")
    assert next(stream) == "Partial answer"
    with pytest.raises(RuntimeError, match="Response stream was interrupted"):
        next(stream)


def test_readiness_requires_selected_google_key(api, monkeypatch):
    main, _client, _auth = api
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("AGENTROUTER_API_KEY", "backup-only")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(main, "_get_store", lambda: type("Store", (), {"collection": type("Collection", (), {"count": lambda self: 0})()})())
    assert main.health()["llm"]["configured"] is False
    with pytest.raises(main.HTTPException) as error:
        main.health_ready()
    assert error.value.status_code == 503
    assert error.value.detail["checks"]["llm_configured"] is False


def test_agentrouter_timeout_error_is_actionable():
    from src.search import RAGSearch

    rag = RAGSearch.__new__(RAGSearch)
    rag.provider = "agentrouter"
    rag.llm_model = "primary-model"

    message = rag._provider_failure_message(RuntimeError("SSL handshake timed out"), streaming=True)
    assert message == "AgentRouter timed out before it could start the response. Please try again."


def test_prompt_text_repairs_mojibake_currency_symbols():
    from src.search import RAGSearch

    assert RAGSearch._normalize_text_for_prompt("face value â\x82¹5 each") == "face value ₹5 each"
    assert RAGSearch._normalize_text_for_prompt("face value â‚¹5 each") == "face value ₹5 each"


def test_source_title_repairs_mojibake_en_dash():
    from src.search import RAGSearch

    assert RAGSearch._repair_mojibake("Updated Draft Red Herring Prospectus â\x80\x93 I") == "Updated Draft Red Herring Prospectus – I"
    assert RAGSearch._repair_mojibake("Updated Draft Red Herring Prospectus â€“ I") == "Updated Draft Red Herring Prospectus – I"


def test_query_embedding_retries_on_cpu_after_mps_oom(monkeypatch):
    import numpy as np
    from src.chroma_store import ChromaVectorStore

    class FakeModel:
        def __init__(self):
            self.calls = []
            self.moved_to = None

        def encode(self, _texts, device=None):
            self.calls.append(device)
            if device == "mps":
                raise RuntimeError("MPS backend out of memory")
            return np.array([[0.1, 0.2]], dtype=np.float32)

        def to(self, device):
            self.moved_to = device

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.model = FakeModel()
    store.embedding_device = "mps"
    monkeypatch.setattr(store, "search", lambda embedding, *_args: embedding)

    assert store.query("question", 5, "alice") == [[0.10000000149011612, 0.20000000298023224]]
    assert store.model.calls == ["mps", "cpu"]
    assert store.model.moved_to == "cpu"
    assert store.embedding_device == "cpu"


def test_reranker_uses_configured_safe_device(monkeypatch):
    from src.search import RAGSearch

    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model, device=None):
            captured.update(model=model, device=device)

    monkeypatch.setattr("src.search.CrossEncoder", FakeCrossEncoder)
    rag = RAGSearch.__new__(RAGSearch)
    rag.reranker = None
    rag._reranker_load_failed = False
    rag.reranker_model = "test-reranker"
    rag.reranker_device = "cpu"

    assert isinstance(rag._ensure_reranker(), FakeCrossEncoder)
    assert captured == {"model": "test-reranker", "device": "cpu"}


def test_long_sentence_is_bounded():
    from src.embedding import EmbeddingPipeline

    pipeline = EmbeddingPipeline.__new__(EmbeddingPipeline)
    pipeline.chunk_size = 100
    pipeline.chunk_overlap = 10
    pipeline.semantic_threshold = 0.55
    pipeline.semantic_min_chars = 30
    chunks = pipeline._semantic_child_chunks("a" * 350)
    assert len(chunks) > 1
    assert max(map(len, chunks)) <= 100


def test_url_validation_rejects_internal_hosts():
    from src.ingest import validate_url

    for url in ("http://localhost/", "http://127.0.0.1/", "http://169.254.169.254/latest/meta-data"):
        with pytest.raises(ValueError):
            validate_url(url)


def test_docx_fixture_and_blank_files():
    from pathlib import Path
    from src.ingest import load_uploaded_file

    fixture = Path(__file__).parent / "fixtures" / "sample.docx"
    docs = load_uploaded_file("Sample.DOCX", fixture.read_bytes())
    assert docs
    assert "DOCX extraction fixture" in docs[0].page_content
    assert docs[0].metadata["source"] == "Sample.DOCX"
    assert load_uploaded_file("blank.txt", b"") == []


def test_logout_requires_csrf_and_allowed_origin(api):
    _, client, auth = api
    alice = auth("alice")
    assert client.post("/auth/logout", cookies=alice["cookies"]).status_code == 403
    assert client.post("/auth/logout", **alice).status_code == 200


def test_password_registration_login_and_duplicate_email(api):
    _, client, _ = api
    headers = {"Origin": "http://localhost:5173"}
    payload = {
        "name": "Local Person",
        "email": "Local.Person@Example.com",
        "password": "StrongPassword1!",
        "terms_accepted": True,
    }
    registered = client.post("/auth/register", json=payload, headers=headers)
    assert registered.status_code == 200
    assert registered.json()["user"]["provider"] == "password"
    assert client.get("/auth/me").json()["authenticated"] is True

    duplicate = client.post("/auth/register", json=payload, headers=headers)
    assert duplicate.status_code == 409

    client.cookies.clear()
    bad_login = client.post("/auth/login", json={"email": payload["email"], "password": "wrong"}, headers=headers)
    assert bad_login.status_code == 401
    good_login = client.post("/auth/login", json={"email": payload["email"], "password": payload["password"]}, headers=headers)
    assert good_login.status_code == 200
    assert client.get("/auth/me").json()["user"]["email"] == "local.person@example.com"


def test_password_auth_requires_trusted_origin_and_strong_password(api):
    _, client, _ = api
    payload = {"name": "Origin Check", "email": "origin@example.com", "password": "StrongPassword1!", "terms_accepted": True}
    assert client.post("/auth/register", json=payload).status_code == 403
    weak = {**payload, "email": "weak@example.com", "password": "password"}
    assert client.post("/auth/register", json=weak, headers={"Origin": "http://localhost:5173"}).status_code == 422


def test_synthetic_ingestion_to_scoped_cited_response(tmp_path, monkeypatch):
    import numpy as np
    import src.chroma_store as chroma_module
    import src.embedding as embedding_module
    from src.search import RAGSearch

    class FakeEmbeddings:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            rows = []
            for text in texts:
                value = str(text).casefold()
                rows.append([
                    float(value.count("alpha")),
                    float(value.count("beta")),
                    float(len(value) % 17) / 17,
                    1.0,
                ])
            return np.asarray(rows, dtype=float)

    monkeypatch.setattr(chroma_module, "SentenceTransformer", FakeEmbeddings)
    monkeypatch.setattr(embedding_module, "SentenceTransformer", FakeEmbeddings)

    store = chroma_module.ChromaVectorStore(persist_dir=str(tmp_path / "chroma"), chunk_size=160, chunk_overlap=20)
    first = store.add_documents(
        [Document(page_content="Alpha policy grants sixteen weeks of leave.", metadata={"source": "Policy.TXT", "source_type": "upload"})],
        user_id="alice",
    )
    store.add_documents(
        [Document(page_content="Beta policy belongs to Bob only.", metadata={"source": "private.txt", "source_type": "upload"})],
        user_id="bob",
    )
    source_id = first[0]["source_id"]

    rag = RAGSearch.__new__(RAGSearch)
    rag.vectorstore = store
    rag.reranker = None
    rag._reranker_load_failed = True
    rag._bm25_cache = {}
    rag._invoke_with_fallback = lambda *_args, **_kwargs: type("Response", (), {"content": "The policy grants sixteen weeks [1]."})()

    response = rag.answer_with_sources("What does the alpha policy grant?", user_id="alice", source_ids=[source_id], use_reranker=False)
    assert response["answer"].endswith("[1].")
    assert response["sources"]
    assert {item["source"] for item in response["sources"]} == {"Policy.TXT"}

    # A changed source replaces its previous chunks without creating duplicates.
    store.add_documents(
        [Document(page_content="Alpha policy now grants eighteen weeks.", metadata={"source": "Policy.TXT", "source_type": "upload"})],
        user_id="alice",
    )
    rows = store.get_user_metadata("alice", [source_id])
    assert len(rows) == 1
    assert "eighteen weeks" in rows[0]["text"]
