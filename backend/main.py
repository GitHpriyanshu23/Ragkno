"""
backend/main.py

FastAPI backend exposing RAG pipeline and Google Drive OAuth endpoints.
"""

import os
import uuid
import json
import pickle
import base64
import hashlib
import hmac
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel

# Always load env relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Allow OAuth over HTTP on localhost during development
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

# Patch sys.path so we can import from `src/`
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drive_loader import (
    disconnect,
    exchange_code,
    get_auth_url,
    is_connected,
    list_drive_files,
    load_credentials,
    load_documents_from_drive,
    pop_oauth_verifier,
    save_oauth_verifier,
)
from src.chroma_store import ChromaVectorStore
from src.database import Database
from src.chat_memory import ChatMemoryStore
from src.ingest import load_uploaded_files, load_url_content

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

STORE_DIR = str(Path(__file__).parent.parent / "chroma_store")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")
SESSION_COOKIE_NAME = "ragkno_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
APP_LOGIN_REDIRECT_URI = os.getenv("GOOGLE_APP_REDIRECT_URI", "http://localhost:8000/login/google/callback")
SESSION_SECRET = os.getenv("RAGKNO_SESSION_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET") or "ragkno-dev-session-secret"

app = FastAPI(title="RAG API", version="1.0.0")

_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
if FRONTEND_URL and FRONTEND_URL not in _allowed_origins:
    _allowed_origins.append(FRONTEND_URL)

_extra_origins = os.getenv("CORS_ORIGINS", "")
if _extra_origins:
    for _o in _extra_origins.split(","):
        _o = _o.strip()
        if _o and _o not in _allowed_origins:
            _allowed_origins.append(_o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_db = Database.get_instance()
_store_instance = None

def _get_store() -> ChromaVectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaVectorStore(persist_dir=STORE_DIR)
    return _store_instance



def _build_login_flow() -> Flow:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth credentials are not configured.")

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": [APP_LOGIN_REDIRECT_URI],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        ],
        redirect_uri=APP_LOGIN_REDIRECT_URI,
    )


def _sign_payload(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    signature = hmac.new(SESSION_SECRET.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _verify_session_token(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    expected = hmac.new(SESSION_SECRET.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        padded = encoded + ("=" * (-len(encoded) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))
    except Exception:
        return None
    if int(payload.get("exp", 0) or 0) < int(time.time()):
        return None
    return payload


def _session_user_from_request(request: Request) -> dict | None:
    payload = _verify_session_token(request.cookies.get(SESSION_COOKIE_NAME))
    if not payload:
        return None
    return {
        "id": payload.get("sub"),
        "email": payload.get("email"),
        "name": payload.get("name") or payload.get("email") or "RagKno user",
        "picture": payload.get("picture"),
    }


def _is_prod_cookie() -> bool:
    return FRONTEND_URL.startswith("https://") or os.getenv("ENV", "").lower() in {"production", "prod"}


def _set_session_cookie(response: Response, user_info: dict) -> None:
    now = int(time.time())
    token = _sign_payload({
        "sub": user_info.get("sub"),
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
        "iat": now,
        "exp": now + SESSION_TTL_SECONDS,
    })
    is_prod = _is_prod_cookie()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="none" if is_prod else "lax",
        secure=is_prod,
        path="/",
    )


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------


@app.get("/app-auth/google/url", summary="Get Google app-login consent URL")
@app.get("/login/google/url", summary="Get Google app-login consent URL")
def login_google_url():
    flow = _build_login_flow()
    auth_url, state = flow.authorization_url(
        access_type="online",
        include_granted_scopes="true",
        prompt="select_account",
    )
    if not flow.code_verifier:
        raise HTTPException(status_code=500, detail="Failed to initialize login verifier.")
    save_oauth_verifier(state, flow.code_verifier)
    return {"url": auth_url}


@app.get("/login/google/callback", summary="Google app-login callback")
def login_google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    if error:
        raise HTTPException(
            status_code=400,
            detail=f"Google login error: {error}. {error_description or ''}".strip(),
        )
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code in login callback.")

    flow = _build_login_flow()
    if state:
        code_verifier = pop_oauth_verifier(state)
        if code_verifier:
            flow.code_verifier = code_verifier

    try:
        flow.fetch_token(code=code)
        token = flow.credentials.id_token
        if not token:
            raise RuntimeError("Google did not return an identity token.")
        user_info = id_token.verify_oauth2_token(
            token,
            GoogleAuthRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
        )
        try:
            _db.upsert_user(user_info)
        except Exception as db_err:
            print(f"[WARN] Failed to upsert user to DB: {db_err}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google login failed: {e}") from e

    response = RedirectResponse(url=f"{FRONTEND_URL}/chat")
    _set_session_cookie(response, user_info)
    return response


@app.get("/auth/me", summary="Current logged-in app user")
def auth_me(request: Request):
    user = _session_user_from_request(request)
    if not user:
        return {"authenticated": False, "user": None}
    user_dict = dict(user)
    if user.get("picture"):
        user_dict["avatar_url"] = "/auth/avatar"
    return {"authenticated": True, "user": user_dict}


@app.get("/auth/avatar", summary="Proxy active user Google avatar image")
async def auth_avatar(request: Request, url: str | None = None):
    target_url = None
    if url and url.startswith("https://lh3.googleusercontent.com/"):
        target_url = url
    else:
        user = _session_user_from_request(request)
        if user and user.get("picture"):
            target_url = user["picture"]

    if not target_url:
        raise HTTPException(status_code=404, detail="No avatar found.")

    import urllib.request
    try:
        req = urllib.request.Request(
            target_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read()
            media_type = resp.headers.get("content-type", "image/jpeg")
            return Response(
                content=content,
                media_type=media_type,
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Access-Control-Allow-Origin": "*",
                },
            )
    except Exception as e:
        return RedirectResponse(url=target_url)



@app.post("/auth/logout", summary="Log out current app user")
def auth_logout(response: Response):
    is_prod = _is_prod_cookie()
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        samesite="none" if is_prod else "lax",
        secure=is_prod,
    )
    return {"ok": True, "message": "Logged out."}



# ---------------------------------------------------------------------------
# Thread & Message routes (per-user ChatGPT-style chat management)
# ---------------------------------------------------------------------------


class CreateThreadRequest(BaseModel):
    title: str = "New Chat"
    id: str | None = None


class RenameThreadRequest(BaseModel):
    title: str


@app.get("/threads", summary="Get all chat threads for the authenticated user")
def get_threads(request: Request):
    user = _session_user_from_request(request)
    if not user:
        return {"threads": []}
    threads = _db.get_user_threads(user["id"])
    return {"threads": threads}


@app.post("/threads", summary="Create a new chat thread")
def create_thread(req: CreateThreadRequest, request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else "guest"
    thread = _db.create_thread(user_id=user_id, title=req.title, thread_id=req.id)
    return {"thread": thread}


@app.get("/threads/{thread_id}/messages", summary="Get all messages for a thread")
def get_thread_messages(thread_id: str, request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    messages = _db.get_thread_messages(thread_id, user_id=user_id)
    return {"messages": messages}


@app.patch("/threads/{thread_id}", summary="Rename a thread")
def rename_thread(thread_id: str, req: RenameThreadRequest, request: Request):
    user = _session_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    ok = _db.rename_thread(thread_id, user["id"], req.title)
    return {"ok": ok}


@app.delete("/threads/{thread_id}", summary="Delete a thread and its messages")
def delete_thread(thread_id: str, request: Request):
    user = _session_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    ok = _db.delete_thread(thread_id, user["id"])
    return {"ok": ok}


@app.get("/auth/url", summary="Get Google OAuth consent URL")
def auth_url(request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    url = get_auth_url(user_id=user_id)
    return {"url": url}


@app.get("/auth/callback", summary="OAuth callback — exchange code and redirect")
def auth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    # If Google sends an explicit OAuth error, return it directly for easier debugging.
    if error:
        raise HTTPException(
            status_code=400,
            detail=f"Google OAuth error: {error}. {error_description or ''}".strip(),
        )

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code in callback.")

    user = _session_user_from_request(request)
    user_id = user["id"] if user else None

    try:
        exchange_code(code, state=state, user_id=user_id)
    except Exception as e:
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")
        raise HTTPException(
            status_code=400,
            detail=(
                "Token exchange failed. Verify OAuth client settings, tester access, "
                f"and redirect URI ({redirect_uri}). Raw error: {e}"
            ),
        )
    # Redirect browser back to the React app with a success flag
    return RedirectResponse(url=f"{FRONTEND_URL}/chat/data?connected=1")


@app.get("/auth/status", summary="Check if Drive is connected for active user")
def auth_status(request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    if not user_id:
        return {"connected": False}
    return {"connected": is_connected(user_id)}


# ---------------------------------------------------------------------------
# Drive routes
# ---------------------------------------------------------------------------


@app.get("/drive/files", summary="List PDF/TXT files in Google Drive for active user")
def drive_files(request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    creds = load_credentials(user_id)
    if not creds:
        raise HTTPException(status_code=401, detail="Not connected to Google Drive.")
    try:
        files = list_drive_files(creds)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DriveSyncRequest(BaseModel):
    file_ids: list[str] | None = None


class UrlIngestRequest(BaseModel):
    url: str


class UnindexRequest(BaseModel):
    key: str


def _load_index_metadata() -> list[dict]:
    try:
        return _get_store().metadata
    except Exception:
        return []


def _source_type(source: str) -> str:
    lower = source.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        return "url"
    if lower.startswith("drive://") or "drive.google.com" in lower:
        return "drive"
    return "upload"


@app.get("/ingest/sources", summary="List unique indexed sources for active user")
def ingest_sources(request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    if not user_id:
        return {"count": 0, "sources": []}

    db_sources = _db.get_user_sources(user_id)
    chroma_sources = _get_store().get_user_sources(user_id=user_id)

    seen = set()
    sources = []

    for s in db_sources:
        k = str(s.get("source_key") or "").lower().rstrip("/")
        if k and k not in seen:
            seen.add(k)
            sources.append({
                "key": k,
                "source": s.get("source_name") or k,
                "type": s.get("source_type") or _source_type(k),
            })

    for s in chroma_sources:
        k = str(s.get("source") or "").lower().rstrip("/")
        if k and k not in seen:
            seen.add(k)
            sources.append({
                "key": k,
                "source": s.get("title") or k,
                "type": s.get("source_type") or _source_type(k),
            })

    return {"count": len(sources), "sources": sources}


@app.post("/ingest/unindex", summary="Remove one indexed source from Chroma store")
def ingest_unindex(req: UnindexRequest, request: Request):
    key = str(req.key or "").strip().lower().rstrip("/")
    if not key:
        raise HTTPException(status_code=400, detail="Source key is required.")

    user = _session_user_from_request(request)
    user_id = user["id"] if user else None

    try:
        store = _get_store()
        removed_chunks = store.remove_source(key, user_id=user_id)
        if user_id:
            _db.delete_user_source(user_id, key)

        global _rag_search_instance
        _rag_search_instance = None

        return {
            "ok": True,
            "removed": removed_chunks,
            "message": f"Unindexed source successfully ({removed_chunks} chunk(s) removed).",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drive/sync", summary="Ingest selected Drive files into Chroma vector store")
def drive_sync(request: Request, req: DriveSyncRequest | None = None):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")

    creds = load_credentials(user_id)
    if not creds:
        raise HTTPException(status_code=401, detail="Not connected to Google Drive.")
    try:
        selected_file_ids = req.file_ids if req else None
        docs = load_documents_from_drive(creds, file_ids=selected_file_ids)
        if not docs:
            if selected_file_ids:
                return {
                    "message": "No extractable content found in the selected files.",
                    "count": 0,
                }
            return {"message": "No supported documents found in Drive.", "count": 0}

        store = _get_store()
        store.add_documents(docs, user_id=user_id)

        for doc in docs:
            src = doc.metadata.get("source", "Google Drive File")
            _db.record_user_source(user_id, source_key=src, source_name=src, source_type="drive", chunk_count=1)

        # Invalidate cache so it reloads the new index on next query
        global _rag_search_instance
        _rag_search_instance = None

        if selected_file_ids is not None:
            return {
                "message": f"Synced {len(docs)} document pages from selected files into the vector store.",
                "count": len(docs),
                "selected_files": len(selected_file_ids),
            }

        return {"message": f"Synced {len(docs)} document pages into the vector store.", "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/drive/disconnect", summary="Disconnect Google Drive for active user")
def drive_disconnect(request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    disconnect(user_id)
    return {"message": "Disconnected from Google Drive."}


@app.post("/ingest/files", summary="Upload local files and index into Chroma")
async def ingest_files(request: Request, files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    payloads = []
    for file in files:
        content = await file.read()
        payloads.append((file.filename or "uploaded-file", content))

    try:
        docs = load_uploaded_files(payloads)
        if not docs:
            return {"ok": True, "message": "No extractable content found.", "indexed_chunks": 0, "source_count": 0}

        user = _session_user_from_request(request)
        user_id = user["id"] if user else "guest"

        store = _get_store()
        store.add_documents(docs, user_id=user_id)

        for file in files:
            fname = file.filename or "uploaded-file"
            _db.record_user_source(user_id, source_key=fname, source_name=fname, source_type="file", chunk_count=len(docs))

        global _rag_search_instance
        _rag_search_instance = None

        return {
            "ok": True,
            "message": f"Indexed content from {len(files)} uploaded file(s).",
            "indexed_chunks": len(docs),
            "source_count": len(files),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url", summary="Fetch and index one website URL")
def ingest_url(req: UrlIngestRequest, request: Request):
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    try:
        docs = load_url_content(req.url.strip())
        user = _session_user_from_request(request)
        user_id = user["id"] if user else "guest"

        store = _get_store()
        store.add_documents(docs, user_id=user_id)

        _db.record_user_source(user_id, source_key=req.url.strip(), source_name=req.url.strip(), source_type="url", chunk_count=len(docs))

        global _rag_search_instance
        _rag_search_instance = None

        return {
            "ok": True,
            "message": "URL content indexed successfully.",
            "indexed_chunks": len(docs),
            "source_count": 1,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Query route
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str | None = None
    thread_id: str | None = None
    model: str | None = None
    use_reranker: bool = True
    language: str | None = None


# Global cache for RAGSearch
_rag_search_instance = None
_memory_store = ChatMemoryStore()


class SessionRequest(BaseModel):
    session_id: str


def _source_link(item: dict) -> str | None:
    source = str(item.get("source", "") or "")
    source_type = str(item.get("type", "") or "").lower()
    file_id = str(item.get("file_id", "") or "")

    if source_type == "url" and source:
        return source

    if source_type == "drive" and file_id:
        return f"https://drive.google.com/file/d/{file_id}/view"

    return None


def _prepare_sources_for_client(raw_sources: list[dict]) -> list[dict]:
    prepared = []
    for item in raw_sources or []:
        prepared.append(
            {
                "index": int(item.get("index", 0) or 0),
                "source": str(item.get("source", "") or ""),
                "type": str(item.get("type", "unknown") or "unknown"),
                "preview": str(item.get("preview", "") or ""),
                "text": str(item.get("text", "") or ""),
                "score": float(item.get("score", 0.0) or 0.0),
                "page": item.get("page"),
                "title": str(item.get("title", "") or ""),
                "file_id": item.get("file_id"),
                "link": _source_link(item),
            }
        )
    return prepared


def _maybe_refresh_summary(session_id: str, rag_search_instance, user_id: str = "default") -> None:
    turns_total = _memory_store.get_turn_count(session_id, user_id=user_id)
    if turns_total < 12 or turns_total % 4 != 0:
        return

    summary_record = _memory_store.get_summary_record(session_id, user_id=user_id)
    unsummarized = _memory_store.get_turns_since(session_id, summary_record["last_turn_id"], user_id=user_id)
    if len(unsummarized) <= 8:
        return

    summarize_now = unsummarized[:-6]
    if not summarize_now:
        return

    history_text = "\n".join(f"{item['role'].title()}: {item['text']}" for item in summarize_now)
    updated_summary = rag_search_instance.summarize_history(
        summary_record["summary_text"],
        history_text,
    )
    _memory_store.upsert_summary(session_id, updated_summary, summarize_now[-1]["id"], user_id=user_id)


def _sse_event(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/memory/reset", summary="Reset chat memory for a session")
def reset_memory(req: SessionRequest, request: Request):
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    user = _session_user_from_request(request)
    user_id = user["id"] if user else "default"
    _memory_store.clear_session(req.session_id.strip(), user_id=user_id)
    return {"ok": True, "message": "Session memory cleared."}


@app.post("/query", summary="RAG query — retrieve and generate answer")
def handle_query(req: QueryRequest, request: Request):
    global _rag_search_instance

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        from src.search import RAGSearch
        if _rag_search_instance is None:
            _rag_search_instance = RAGSearch(persist_dir=STORE_DIR)

        user = _session_user_from_request(request)
        user_id = user["id"] if user else None
        session_id = (req.thread_id or req.session_id or "").strip() or f"anon-{uuid.uuid4().hex[:12]}"
        uid = user_id or "default"

        memory_context = _memory_store.build_memory_context(session_id=session_id, recent_limit=8, user_id=uid)

        response_payload = _rag_search_instance.answer_with_sources(
            query=req.query.strip(),
            top_k=req.top_k,
            memory_context=memory_context,
            user_id=user_id,
            model=req.model,
            use_reranker=req.use_reranker,
        )
        answer = response_payload.get("answer", "")
        sources = _prepare_sources_for_client(response_payload.get("sources", []))

        _memory_store.append_turn(session_id, "user", req.query.strip(), user_id=uid)
        _memory_store.append_turn(session_id, "assistant", answer, user_id=uid)
        _maybe_refresh_summary(session_id, _rag_search_instance, user_id=uid)

        if user_id:
            _db.append_message(thread_id=session_id, user_id=user_id, role="user", text_content=req.query.strip())
            _db.append_message(thread_id=session_id, user_id=user_id, role="assistant", text_content=answer, sources=sources)

        return {
            "answer": answer,
            "query": req.query,
            "session_id": session_id,
            "sources": sources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", summary="RAG query with server-sent events")
def handle_query_stream(req: QueryRequest, request: Request):
    global _rag_search_instance

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    from src.search import RAGSearch
    if _rag_search_instance is None:
        _rag_search_instance = RAGSearch(persist_dir=STORE_DIR)

    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    session_id = (req.thread_id or req.session_id or "").strip() or f"anon-{uuid.uuid4().hex[:12]}"
    uid = user_id or "default"

    memory_context = _memory_store.build_memory_context(session_id=session_id, recent_limit=8, user_id=uid)

    def event_generator():
        try:
            sources, token_iter = _rag_search_instance.stream_answer_with_sources(
                query=req.query.strip(),
                top_k=req.top_k,
                memory_context=memory_context,
                user_id=user_id,
                model=req.model,
                use_reranker=req.use_reranker,
            )
            prepared_sources = _prepare_sources_for_client(sources)
            yield _sse_event(
                "meta",
                {
                    "query": req.query,
                    "session_id": session_id,
                    "sources": prepared_sources,
                },
            )

            answer_parts = []
            for token in token_iter:
                text = str(token or "")
                if not text:
                    continue
                answer_parts.append(text)
                yield _sse_event("token", {"token": text})

            answer = "".join(answer_parts).strip() or "No relevant documents found."
            _memory_store.append_turn(session_id, "user", req.query.strip(), user_id=uid)
            _memory_store.append_turn(session_id, "assistant", answer, user_id=uid)
            _maybe_refresh_summary(session_id, _rag_search_instance, user_id=uid)

            if user_id:
                _db.append_message(thread_id=session_id, user_id=user_id, role="user", text_content=req.query.strip())
                _db.append_message(thread_id=session_id, user_id=user_id, role="assistant", text_content=answer, sources=prepared_sources)

            yield _sse_event(
                "done",
                {
                    "query": req.query,
                    "session_id": session_id,
                    "answer": answer,
                    "sources": prepared_sources,
                },
            )
        except Exception as e:
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ---------------------------------------------------------------------------
# Feedback routes
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    rating: str
    feedback: str


@app.post("/feedback", summary="Submit user feedback")
def submit_feedback(payload: FeedbackRequest, request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    if not payload.feedback.strip() or not payload.rating.strip():
        raise HTTPException(status_code=400, detail="Rating and feedback cannot be empty.")
    saved = _db.save_feedback(user_id=user_id, rating=payload.rating, feedback=payload.feedback)
    return {"ok": True, "feedback": saved}


@app.get("/feedback", summary="List tracked feedback")
def list_feedbacks_route(request: Request, limit: int = 50):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    # Return all feedbacks so developer/admin can track user feedback
    feedbacks = _db.list_feedbacks(user_id=None, limit=min(limit, 100))
    return {"ok": True, "feedbacks": feedbacks}



@app.get("/feedback", summary="List tracked feedback")
def list_feedbacks_route(request: Request, limit: int = 50):
    feedbacks = _db.list_feedbacks(user_id=None, limit=min(limit, 100))
    return {"ok": True, "feedbacks": feedbacks}


# ---------------------------------------------------------------------------
# Feedback routes
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    rating: str
    feedback: str


@app.post("/feedback", summary="Submit user feedback")
def submit_feedback(payload: FeedbackRequest, request: Request):
    user = _session_user_from_request(request)
    user_id = user["id"] if user else None
    if not payload.feedback.strip() or not payload.rating.strip():
        raise HTTPException(status_code=400, detail="Rating and feedback cannot be empty.")
    saved = _db.save_feedback(user_id=user_id, rating=payload.rating, feedback=payload.feedback)
    return {"ok": True, "feedback": saved}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}



if __name__ == "__main__":
    import uvicorn
    # This allows running via `python backend/main.py` and guarantees the local .venv is used
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
