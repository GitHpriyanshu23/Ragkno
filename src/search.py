import os
import re
import math
import json
import time
from collections.abc import Iterator

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

from src.chroma_store import ChromaVectorStore

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "chroma_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "deepseek-v4-flash"):
        self.vectorstore = ChromaVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
        print(f"[INFO] RAGSearch using ChromaVectorStore at '{persist_dir}'")

        self.agentrouter_api_key = os.getenv("AGENTROUTER_API_KEY") or os.getenv("AGENT_ROUTER_API_KEY")
        self.agentrouter_base_url = os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        # Provider latency can spike before the first streamed token. Keep the
        # connect window short, but never inherit an impractically small read
        # timeout from a parent shell or deployment environment.
        self.connect_timeout_seconds = max(3.0, float(os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "10")))
        self.request_timeout_seconds = max(30.0, float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "45")))

        provider_pref = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        if provider_pref not in {"", "agentrouter", "google"}:
            raise ValueError("LLM_PROVIDER must be either 'agentrouter' or 'google'.")

        if provider_pref == "agentrouter" and not self.agentrouter_api_key:
            raise ValueError("LLM_PROVIDER is agentrouter but AGENTROUTER_API_KEY is not configured.")
        if provider_pref == "google" and not self.google_api_key:
            raise ValueError("LLM_PROVIDER is google but GOOGLE_API_KEY is not configured.")

        if provider_pref == "google":
            self.provider = "google"
            self.llm_model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")
            fallback_config = os.getenv("GOOGLE_LLM_FALLBACK_MODELS", "")
            self.fallback_models = [item.strip() for item in fallback_config.split(",") if item.strip()]
            self.llm = ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model=self.llm_model)
            print(f"[INFO] Google LLM initialized: {self.llm_model}")
        elif self.agentrouter_api_key:
            self.provider = "agentrouter"
            self.llm_model = os.getenv("AGENTROUTER_MODEL", llm_model)
            self.agentrouter_headers = {
                "User-Agent": "codex_cli_rs/0.1.0",
                "x-app": "cli",
            }
            self.llm = ChatOpenAI(
                api_key=self.agentrouter_api_key,
                base_url=self.agentrouter_base_url,
                model=self.llm_model,
                streaming=True,
                timeout=self.request_timeout_seconds,
                max_retries=0,
                default_headers=self.agentrouter_headers,
            )
            fallback_config = os.getenv("AGENTROUTER_FALLBACK_MODELS", "")
            self.fallback_models = [item.strip() for item in fallback_config.split(",") if item.strip()]
            print(f"[INFO] AgentRouter LLM initialized: {self.llm_model} at {self.agentrouter_base_url}")
        elif self.google_api_key:
            self.provider = "google"
            self.llm_model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")
            fallback_config = os.getenv("GOOGLE_LLM_FALLBACK_MODELS", "")
            self.fallback_models = [item.strip() for item in fallback_config.split(",") if item.strip()]
            self.llm = ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model=self.llm_model)
            print(f"[INFO] Google LLM initialized: {self.llm_model}")
        else:
            raise ValueError(
                "Neither AGENTROUTER_API_KEY nor GOOGLE_API_KEY is configured in your .env file."
            )

        # Optional semantic reranker for domain-agnostic retrieval quality.
        self.reranker_model = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker = None
        self._reranker_load_failed = False

        # Hybrid retrieval internals (dense + sparse BM25)
        self._bm25_cache: dict[tuple[str, tuple[str, ...]], tuple[object, list[dict]]] = {}
        allowed_config = os.getenv("RAG_ALLOWED_MODELS", "")
        configured_allowed = [item.strip() for item in allowed_config.split(",") if item.strip()]
        self.allowed_models = frozenset([self.llm_model, *self.fallback_models, *configured_allowed])

    def _build_llm(self, model_name: str):
        if model_name not in self.allowed_models:
            raise ValueError("Requested model is not enabled by the server")
        if self.provider == "agentrouter":
            return ChatOpenAI(
                api_key=self.agentrouter_api_key,
                base_url=self.agentrouter_base_url,
                model=model_name,
                streaming=True,
                timeout=self.request_timeout_seconds,
                max_retries=0,
                default_headers=getattr(self, "agentrouter_headers", {
                    "User-Agent": "codex_cli_rs/0.1.0",
                    "x-app": "cli",
                }),
            )
        return ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model=model_name)

    def _model_candidates(self, requested_model: str | None = None) -> list[str]:
        primary = (requested_model or self.llm_model).strip()
        if primary not in self.allowed_models:
            raise ValueError("Requested model is not enabled by the server")
        return [primary, *[model for model in self.fallback_models if model != primary]]

    @staticmethod
    def _should_fallback_for_error(error_text: str) -> bool:
        lower = str(error_text or "").lower()
        return any(
            key in lower for key in [
                "not_found",
                "is not found",
                "resource_exhausted",
                "budget pool",
                "exhausted",
                "unauthorized client",
                "401",
                "402",
                "404",
                "429",
                "quota",
                "rate limit",
                "timed out",
                "timeout",
                "overloaded",
                "server_error",
                "500",
                "502",
                "503",
                "504",
                "connection",
                "failed to resolve",
                "name resolution",
                "ssl",
                "handshake",
                "connection reset",
            ]
        )

    def _provider_failure_message(self, error: Exception | None, *, streaming: bool) -> str:
        detail = str(error or "").lower()
        action = "stream" if streaming else "request"
        if self.provider == "agentrouter":
            if "invalid api key" in detail or "authentication" in detail or "401" in detail:
                return "AgentRouter authentication failed. Update AGENTROUTER_API_KEY with an active key."
            if "model" in detail and ("not found" in detail or "404" in detail):
                return f"AgentRouter model '{self.llm_model}' is not available for this API key."
            if any(marker in detail for marker in ("429", "rate limit", "quota", "budget pool", "exhausted")):
                return "AgentRouter is temporarily rate-limited or out of model capacity. Please try again."
            if any(marker in detail for marker in ("timed out", "timeout", "handshake")):
                return "AgentRouter timed out before it could start the response. Please try again."
            if any(marker in detail for marker in ("connection", "failed to resolve", "name resolution", "ssl")):
                return "AgentRouter is temporarily unreachable. Please try again."
            return f"AgentRouter could not complete the {action}."
        return f"Google AI could not complete the {action}."

    def _invoke_with_fallback(self, prompt: str, requested_model: str | None = None):
        candidates = self._model_candidates(requested_model)
        first_error = None
        for index, model in enumerate(candidates):
            try:
                if self.provider == "agentrouter":
                    content = "".join(self._stream_agentrouter_response(prompt, model)).strip()
                    if not content:
                        raise RuntimeError("The model returned an empty completion stream")
                    return AIMessage(content=content)
                return self._build_llm(model).invoke([prompt])
            except Exception as err:
                first_error = first_error or err
                if index == 0 and not self._should_fallback_for_error(str(err)):
                    raise
                print(f"[WARN] Model '{model}' failed: {err}")
        raise RuntimeError(self._provider_failure_message(first_error, streaming=False)) from first_error

    def _stream_with_fallback(self, prompt: str, requested_model: str | None = None) -> Iterator[str]:
        def stream_from_llm(llm) -> Iterator[str]:
            for chunk in llm.stream([prompt]):
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    pieces = []
                    for item in content:
                        if isinstance(item, dict):
                            text = str(item.get("text", ""))
                            if text:
                                pieces.append(text)
                        else:
                            pieces.append(str(item))
                    content = "".join(pieces)
                content = str(content or "")
                if content:
                    yield content

        candidates = self._model_candidates(requested_model)
        first_error = None
        for index, model in enumerate(candidates):
            for attempt in range(2):
                emitted = False
                try:
                    stream = (
                        self._stream_agentrouter_response(prompt, model)
                        if self.provider == "agentrouter"
                        else stream_from_llm(self._build_llm(model))
                    )
                    for piece in stream:
                        emitted = True
                        yield piece
                    if not emitted:
                        raise RuntimeError("The model returned an empty completion stream")
                    return
                except Exception as err:
                    first_error = first_error or err
                    if emitted:
                        raise RuntimeError("Response stream was interrupted") from err
                    transient = self._should_fallback_for_error(str(err))
                    if transient and attempt == 0:
                        print(f"[WARN] Model '{model}' stream attempt failed; retrying once: {err}")
                        time.sleep(0.35)
                        continue
                    if index == 0 and not transient:
                        raise
                    print(f"[WARN] Model '{model}' stream failed: {err}")
                    break
        raise RuntimeError(self._provider_failure_message(first_error, streaming=True)) from first_error

    def _stream_agentrouter_response(self, prompt: str, model: str) -> Iterator[str]:
        """Read AgentRouter SSE directly.

        Some OpenAI-compatible providers send keepalive or terminal chunks with
        ``choices: [null]`` or ``delta: null``. The OpenAI/LangChain adapter used
        by ChatOpenAI currently attempts ``model_dump()`` on those values and
        terminates an otherwise healthy stream. Parsing the small SSE surface
        here lets us ignore non-content events while retaining real provider
        errors and incremental output.
        """
        url = f"{self.agentrouter_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.agentrouter_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **getattr(self, "agentrouter_headers", {}),
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        try:
            with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(self.connect_timeout_seconds, self.request_timeout_seconds),
            ) as response:
                if not response.ok:
                    detail = response.text.strip()[:500]
                    raise RuntimeError(
                        f"AgentRouter returned HTTP {response.status_code}"
                        + (f": {detail}" if detail else "")
                    )

                for raw_line in response.iter_lines(decode_unicode=True):
                    line = str(raw_line or "").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        return
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        error = event["error"]
                        message = error.get("message") if isinstance(error, dict) else str(error)
                        raise RuntimeError(f"AgentRouter stream error: {message or 'unknown error'}")

                    choices = event.get("choices") or []
                    if not isinstance(choices, list):
                        continue
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta") or {}
                        if not isinstance(delta, dict):
                            continue
                        content = delta.get("content")
                        if isinstance(content, str):
                            if content:
                                yield content
                            continue
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    text = item.get("text")
                                    if isinstance(text, str) and text:
                                        yield text
        except requests.Timeout as error:
            raise RuntimeError(
                f"AgentRouter timed out after {self.request_timeout_seconds:g} seconds while waiting for a response"
            ) from error
        except requests.RequestException as error:
            raise RuntimeError(f"AgentRouter stream request failed: {error}") from error

    def _ensure_reranker(self) -> CrossEncoder | None:
        if self.reranker is not None:
            return self.reranker
        if self._reranker_load_failed:
            return None

        try:
            self.reranker = CrossEncoder(self.reranker_model)
            print(f"[INFO] Semantic reranker initialized: {self.reranker_model}")
            return self.reranker
        except Exception as err:
            self._reranker_load_failed = True
            print(f"[WARN] Reranker unavailable ({self.reranker_model}): {err}")
            return None

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        stop = {
            "the", "a", "an", "for", "of", "to", "in", "on", "is", "are",
            "was", "were", "be", "been", "being", "and", "or", "with", "it",
            "this", "that", "he", "she", "they", "what", "where", "when", "who",
            "how", "tell", "me", "about", "had", "have", "has",
        }
        terms = []
        for tok in (query or "").lower().replace("'", " ").split():
            cleaned = "".join(ch for ch in tok if ch.isalnum() or ch in {"_", "-"})
            if cleaned and cleaned not in stop and len(cleaned) >= 3:
                terms.append(cleaned)
        return terms

    @staticmethod
    def _tokenize_for_bm25(text: str) -> list[str]:
        return re.findall(r"[^\W_]+(?:[-_][^\W_]+)*", str(text or "").casefold(), flags=re.UNICODE)

    def _ensure_bm25_index(self, user_id: str, source_ids: list[str] | None = None):
        if BM25Okapi is None:
            return None
        cache_key = (user_id, tuple(sorted(source_ids or [])))
        if cache_key in self._bm25_cache:
            return self._bm25_cache[cache_key]
        rows = self.vectorstore.get_user_metadata(user_id, source_ids)
        if not rows:
            return None

        tokenized = []
        mapped_rows = []
        for row_index, row in enumerate(rows):
            metadata = dict(row or {})
            text = str(metadata.get("text", "") or "")
            tokens = self._tokenize_for_bm25(text)
            if not tokens:
                continue
            tokenized.append(tokens)
            mapped_rows.append(metadata)

        if not tokenized:
            return None
        value = (BM25Okapi(tokenized), mapped_rows)
        self._bm25_cache[cache_key] = value
        return value

    def _bm25_retrieve(self, query: str, top_k: int, user_id: str, source_ids: list[str] | None = None) -> list[dict]:
        prepared = self._ensure_bm25_index(user_id, source_ids)
        if not prepared:
            return []
        index, rows = prepared

        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = index.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

        results = []
        for rank_pos, internal_idx in enumerate(ranked_indices[:top_k], start=1):
            metadata = rows[internal_idx]
            score = float(scores[internal_idx])
            if not math.isfinite(score) or score <= 0:
                continue
            results.append(
                {
                    "index": rank_pos - 1,
                    "id": metadata.get("chunk_id"),
                    "distance": None,
                    "metadata": metadata,
                    "bm25_score": score,
                    "bm25_rank": rank_pos,
                }
            )
        return results

    @staticmethod
    def _rrf_fuse(dense_results: list[dict], sparse_results: list[dict], dense_weight: float = 0.6, sparse_weight: float = 0.4, rrf_k: int = 60) -> list[dict]:
        merged: dict[str, dict] = {}

        for rank, row in enumerate(dense_results, start=1):
            key = str(row.get("id") or (row.get("metadata") or {}).get("chunk_id") or "")
            if not key:
                continue
            payload = dict(row)
            payload["dense_rank"] = rank
            payload["sparse_rank"] = None
            payload["hybrid_score"] = dense_weight * (1.0 / (rrf_k + rank))
            merged[key] = payload

        for rank, row in enumerate(sparse_results, start=1):
            key = str(row.get("id") or (row.get("metadata") or {}).get("chunk_id") or "")
            if not key:
                continue
            if key not in merged:
                payload = dict(row)
                payload["dense_rank"] = None
                payload["distance"] = None
                payload["hybrid_score"] = 0.0
                merged[key] = payload
            merged[key]["sparse_rank"] = rank
            merged[key]["hybrid_score"] += sparse_weight * (1.0 / (rrf_k + rank))
            merged[key]["bm25_score"] = float(row.get("bm25_score", 0.0))

        fused = list(merged.values())
        fused.sort(key=lambda item: float(item.get("hybrid_score", 0.0)), reverse=True)
        return fused

    @staticmethod
    def _lexical_overlap_score(query_terms: list[str], text: str) -> float:
        if not query_terms:
            return 0.0
        lower_text = (text or "").lower()
        hits = sum(1 for term in query_terms if term in lower_text)
        return hits / float(len(query_terms))

    @staticmethod
    def _normalize_text_for_prompt(text: str) -> str:
        # Normalize common PDF ligatures and whitespace artifacts for cleaner reasoning.
        replacements = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "’": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
        }
        value = RAGSearch._repair_mojibake(text or "")
        for old, new in replacements.items():
            value = value.replace(old, new)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    @staticmethod
    def _repair_mojibake(text: str) -> str:
        """Repair UTF-8 text that a PDF/source loader decoded as Latin-1 or CP1252."""
        value = str(text or "")
        suspicious = ("Ã", "Â", "â", "ðŸ", "ï»¿")
        if not any(marker in value for marker in suspicious):
            return value

        def corruption_score(candidate: str) -> int:
            marker_count = sum(candidate.count(marker) for marker in suspicious)
            control_count = sum(1 for char in candidate if 0x80 <= ord(char) <= 0x9F)
            return (marker_count * 4) + (control_count * 3) + candidate.count("�") * 8

        best = value
        best_score = corruption_score(value)
        for encoding in ("latin-1", "cp1252"):
            try:
                candidate = value.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            score = corruption_score(candidate)
            if score < best_score:
                best = candidate
                best_score = score
        return best

    @staticmethod
    def _memory_retrieval_hint(memory_context: str | None) -> str:
        if not memory_context:
            return ""

        lines = [line.strip() for line in memory_context.splitlines() if line.strip()]
        user_lines = [line[5:].strip() for line in lines if line.lower().startswith("user:")]
        summary_lines = [line for line in lines if not line.lower().startswith(("user:", "assistant:"))]

        parts = []
        if summary_lines:
            parts.append(" ".join(summary_lines)[:220])
        if user_lines:
            parts.append(" ".join(user_lines[-2:])[:220])

        return " ".join(p for p in parts if p).strip()

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        return normalized[:300]

    def _diversify_results(self, ranked: list[dict], max_per_source: int = 2) -> list[dict]:
        max_per_source = max(1, int(os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", str(max_per_source))))
        source_counts: dict[str, int] = {}
        seen_fp: set[str] = set()
        diversified: list[dict] = []

        for r in ranked:
            meta = r.get("metadata") or {}
            source = str(meta.get("source", "unknown"))
            text = str(meta.get("text", ""))
            fp = self._fingerprint(text)

            if fp in seen_fp:
                continue
            if source_counts.get(source, 0) >= max_per_source:
                continue

            diversified.append(r)
            seen_fp.add(fp)
            source_counts[source] = source_counts.get(source, 0) + 1

        return diversified

    def invalidate_caches(self) -> None:
        self._bm25_cache.clear()

    def _extract_direct_snippets(self, query: str, results: list[dict], max_snippets: int = 4) -> list[str]:
        terms = [t for t in self._query_terms(query) if len(t) >= 4]
        snippets: list[str] = []

        if not terms:
            return snippets

        for r in results:
            text = self._normalize_text_for_prompt(str((r.get("metadata") or {}).get("text", "") or ""))
            lower = text.lower()
            match_pos = -1
            for term in terms:
                pos = lower.find(term)
                if pos != -1:
                    match_pos = pos
                    break

            if match_pos == -1:
                continue

            start = max(0, match_pos - 140)
            end = min(len(text), match_pos + 240)
            snippet = text[start:end].strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet)

            if len(snippets) >= max_snippets:
                break

        return snippets

    def _rank_results(self, query: str, results: list[dict]) -> list[dict]:
        if not results:
            return results

        reranker = self._ensure_reranker()
        if reranker is None:
            # Fallback to hybrid score first, then vector distance.
            ordered = sorted(
                results,
                key=lambda r: (
                    float(r.get("hybrid_score", 0.0)),
                    -float(r.get("distance", 1e9)) if r.get("distance") not in (None, float("inf")) else -1e9,
                ),
                reverse=True,
            )
            return [{**item, "score": float(item.get("hybrid_score", 0.0))} for item in ordered]

        pairs = []
        for r in results:
            meta = r.get("metadata") or {}
            text = str(meta.get("text", "") or "")
            pairs.append((query, text[:2500]))

        try:
            ce_scores = reranker.predict(pairs)
        except Exception as err:
            print(f"[WARN] Reranker inference failed, using vector order: {err}")
            ordered = sorted(
                results,
                key=lambda r: (
                    float(r.get("hybrid_score", 0.0)),
                    -float(r.get("distance", 1e9)) if r.get("distance") not in (None, float("inf")) else -1e9,
                ),
                reverse=True,
            )
            return [{**item, "score": float(item.get("hybrid_score", 0.0))} for item in ordered]

        query_terms = self._query_terms(query)
        texts_lower = [str((r.get("metadata") or {}).get("text", "") or "").lower() for r in results]
        term_doc_freq = {
            term: sum(1 for text in texts_lower if term in text)
            for term in query_terms
        }

        ranked = []
        for idx, r in enumerate(results):
            raw_distance = r.get("distance")
            distance = float(raw_distance) if raw_distance is not None and math.isfinite(float(raw_distance)) else 1.0
            ce_score = float(ce_scores[idx])
            meta = r.get("metadata") or {}
            text = str(meta.get("text", "") or "")
            lexical = self._lexical_overlap_score(query_terms, text)
            lower_text = text.lower()

            rare_boost = 0.0
            for term in query_terms:
                df = term_doc_freq.get(term, 0)
                if df == 0:
                    continue
                if term in lower_text and df <= 3:
                    rare_boost += 1.8

            # Cross-encoder is primary; lexical overlap helps exact factual terms.
            final_score = ce_score + (0.6 * lexical) + rare_boost - (0.02 * distance)
            ranked.append((final_score, {**r, "score": final_score}))

        ranked.sort(key=lambda x: x[0], reverse=True)
        ordered = [r for _, r in ranked]
        diversified = self._diversify_results(ordered, max_per_source=2)
        return diversified if diversified else ordered

    @staticmethod
    def _expand_to_parent_context(results: list[dict], top_k: int) -> list[dict]:
        expanded = []
        used_parent_ids: set[str] = set()

        for row in results:
            metadata = dict(row.get("metadata") or {})
            parent_id = str(metadata.get("parent_id", "") or "")
            parent_text = str(metadata.get("parent_text", "") or "")

            if parent_id and parent_id in used_parent_ids:
                continue

            if parent_text:
                metadata["child_text"] = str(metadata.get("text", "") or "")
                metadata["text"] = parent_text

            if parent_id:
                used_parent_ids.add(parent_id)

            expanded.append({**row, "metadata": metadata})
            if len(expanded) >= top_k:
                break

        return expanded

    @staticmethod
    def _source_type_from_value(source: str, metadata: dict) -> str:
        source_type = str(metadata.get("source_type", "") or "").strip().lower()
        if source_type:
            return source_type
        lower = source.lower()
        if lower.startswith("http://") or lower.startswith("https://"):
            return "url"
        if lower.startswith("drive://"):
            return "drive"
        return "upload"

    def _build_source_payload(self, results: list[dict]) -> list[dict]:
        sources = []
        for idx, row in enumerate(results, start=1):
            metadata = dict(row.get("metadata") or {})
            text = self._normalize_text_for_prompt(str(metadata.get("text", "") or ""))
            source = self._repair_mojibake(str(metadata.get("source", "Unknown source") or "Unknown source"))
            preview = text[:150] + ("..." if len(text) > 150 else "")
            source_type = self._source_type_from_value(source, metadata)

            sources.append(
                {
                    "index": idx,
                    "source": source,
                    "type": source_type,
                    "preview": preview,
                    "text": text,
                    "score": float(row.get("score", 0.0)) if math.isfinite(float(row.get("score", 0.0))) else 0.0,
                    "distance": float(row.get("distance")) if row.get("distance") is not None and math.isfinite(float(row.get("distance"))) else None,
                    "source_id": metadata.get("source_id"),
                    "file_id": metadata.get("file_id"),
                    "page": metadata.get("page"),
                    "title": self._repair_mojibake(str(metadata.get("title", "") or "")),
                }
            )
        return sources

    def _build_answer_prompt(
        self,
        query: str,
        sources: list[dict],
        direct_snippets: list[str],
        memory_context: str | None,
        language: str | None = None,
    ) -> str:
        context_blocks = []
        for source in sources:
            block = (
                f"[{source['index']}] Source: {source['source']}\n"
                f"Content:\n{source['text']}"
            )
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)
        if not context.strip():
            return ""

        direct_match_block = ""
        if direct_snippets:
            direct_match_block = "Directly matched context snippets:\n" + "\n\n".join(direct_snippets) + "\n\n"

        memory_block = ""
        if memory_context:
            memory_block = f"Conversation memory:\n{memory_context}\n\n"

        language_instruction = ""
        if language and language.lower() not in {"auto", ""}:
            language_instruction = f"Write the answer in language code '{language}'.\n"

        return (
            "Answer the user query using only relevant information from the retrieved context.\n"
            "Whenever you make a factual claim from context, add inline citations like [1], [2].\n"
            "If multiple sources support one claim, cite all relevant references.\n"
            "Do not fabricate citations.\n\n"
            "When presenting comparable multi-row data, use a valid Markdown table with a header and separator row.\n"
            "Use proper Unicode currency symbols (for example ₹) and never reproduce garbled encoding such as â characters.\n"
            f"{language_instruction}"
            f"{memory_block}"
            f"{direct_match_block}"
            f"User query: {query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "If context is insufficient, say what is missing briefly."
        )

    def _retrieve_for_answer(self, query: str, top_k: int, memory_context: str | None = None, user_id: str | None = None, use_reranker: bool = True, source_ids: list[str] | None = None) -> tuple[list[dict], str]:
        if not user_id:
            raise ValueError("Authenticated user_id is required for retrieval")
        top_k = max(1, min(int(top_k), 20))
        query_terms = self._query_terms(query)
        strong_terms = [t for t in query_terms if len(t) >= 7]
        retrieval_hint = self._memory_retrieval_hint(memory_context)
        use_memory_for_retrieval = bool(retrieval_hint) and len(strong_terms) == 0
        retrieval_query = query if not use_memory_for_retrieval else f"{query} {retrieval_hint}"

        candidate_k = max(top_k * 20, 120)
        dense_results = self.vectorstore.query(retrieval_query, top_k=candidate_k, user_id=user_id, source_ids=source_ids)
        sparse_results = self._bm25_retrieve(retrieval_query, top_k=candidate_k, user_id=user_id, source_ids=source_ids)
        fused_results = self._rrf_fuse(dense_results, sparse_results)
        ranked = self._rank_results(retrieval_query, fused_results) if use_reranker else fused_results
        rerank_cutoff = float(os.getenv("RAG_RELEVANCE_CUTOFF", "-2.0"))
        vector_distance_cutoff = float(os.getenv("RAG_MAX_VECTOR_DISTANCE", "0.85"))
        relevant = []
        for row in ranked:
            score = row.get("score")
            distance = row.get("distance")
            bm25_score = float(row.get("bm25_score", 0.0) or 0.0)
            if score is not None and (not math.isfinite(float(score)) or float(score) < rerank_cutoff):
                continue
            if distance is not None and (not math.isfinite(float(distance)) or float(distance) > vector_distance_cutoff) and bm25_score <= 0:
                continue
            relevant.append(row)
        ranked = relevant
        ranked = [row for row in ranked if (row.get("metadata") or {}).get("user_id") == user_id]
        parent_expanded = self._expand_to_parent_context(ranked, top_k=top_k)
        return parent_expanded, retrieval_query

    def answer_with_sources(self, query: str, top_k: int = 5, memory_context: str | None = None, user_id: str | None = None, model: str | None = None, use_reranker: bool = True, source_ids: list[str] | None = None, language: str | None = None, **kwargs) -> dict:
        ranked_results, _ = self._retrieve_for_answer(query, top_k, memory_context, user_id=user_id, use_reranker=use_reranker, source_ids=source_ids)
        ranked_results = [row for row in ranked_results if (row.get("metadata") or {}).get("user_id") == user_id]
        sources = self._build_source_payload(ranked_results)
        if not sources:
            return {"answer": "No relevant documents found.", "sources": []}

        direct_snippets = self._extract_direct_snippets(query, ranked_results)
        prompt = self._build_answer_prompt(query, sources, direct_snippets, memory_context, language)
        response = self._invoke_with_fallback(prompt, model)
        answer = str(response.content or "").strip() or "No relevant documents found."
        return {"answer": answer, "sources": sources}

    def stream_answer_with_sources(self, query: str, top_k: int = 5, memory_context: str | None = None, user_id: str | None = None, model: str | None = None, use_reranker: bool = True, source_ids: list[str] | None = None, language: str | None = None, **kwargs) -> tuple[list[dict], Iterator[str]]:
        ranked_results, _ = self._retrieve_for_answer(query, top_k, memory_context, user_id=user_id, use_reranker=use_reranker, source_ids=source_ids)
        ranked_results = [row for row in ranked_results if (row.get("metadata") or {}).get("user_id") == user_id]
        sources = self._build_source_payload(ranked_results)
        if not sources:
            return [], iter(["No relevant documents found."])

        direct_snippets = self._extract_direct_snippets(query, ranked_results)
        prompt = self._build_answer_prompt(query, sources, direct_snippets, memory_context, language)
        return sources, self._stream_with_fallback(prompt, model)

    def summarize_history(self, existing_summary: str, history_text: str) -> str:
        prompt = (
            "Update the conversation summary with the new turns. Keep important user facts, goals, "
            "constraints, and unresolved questions concise and accurate.\n\n"
            f"Existing summary:\n{existing_summary or 'None'}\n\n"
            f"New turns:\n{history_text}\n\n"
            "Updated summary:"
        )
        response = self._invoke_with_fallback(prompt)
        return (response.content or "").strip()

    def search_and_summarize(self, query: str, top_k: int = 5, memory_context: str | None = None) -> str:
        result = self.answer_with_sources(query=query, top_k=top_k, memory_context=memory_context)
        return result.get("answer", "No relevant documents found.")

# Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What is attention mechanism?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)
