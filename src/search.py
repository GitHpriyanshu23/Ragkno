import os
import re
from collections.abc import Iterator

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

from src.vectorstore import FaissVectorStore

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

load_dotenv()

class RAGSearch:
    FALLBACK_MODELS = [
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemma-3n-e4b-it",
        "gemma-3n-e2b-it",
    ]

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma-4-26b-it"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Add it to your environment or .env file.")

        # Allow changing model from env without code edits.
        self.llm_model = os.getenv("GOOGLE_LLM_MODEL", llm_model)
        self.google_api_key = google_api_key

        self.llm = ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model=self.llm_model)
        print(f"[INFO] Google LLM initialized: {self.llm_model}")

        # Optional semantic reranker for domain-agnostic retrieval quality.
        self.reranker_model = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker = None
        self._reranker_load_failed = False

        # Hybrid retrieval internals (dense + sparse BM25)
        self._bm25_index = None
        self._bm25_tokens = []
        self._bm25_rows = []

    def _try_switch_model(self, model_name: str) -> None:
        self.llm = ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model=model_name)
        self.llm_model = model_name
        print(f"[INFO] Switched Google LLM model to: {model_name}")

    @staticmethod
    def _should_fallback_for_error(error_text: str) -> bool:
        return any(
            key in error_text for key in [
                "NOT_FOUND",
                "is not found",
                "RESOURCE_EXHAUSTED",
                "429",
                "quota",
                "rate limit",
            ]
        )

    def _invoke_with_fallback(self, prompt: str):
        try:
            return self.llm.invoke([prompt])
        except Exception as err:
            text = str(err)
            should_fallback = self._should_fallback_for_error(text)
            if not should_fallback:
                raise

            candidates = [m for m in self.FALLBACK_MODELS if m != self.llm_model]
            print(f"[WARN] Model '{self.llm_model}' failed. Trying fallbacks: {candidates}")

            for model in candidates:
                try:
                    self._try_switch_model(model)
                    return self.llm.invoke([prompt])
                except Exception as fallback_err:
                    print(f"[WARN] Fallback model '{model}' failed: {fallback_err}")

            raise RuntimeError(
                "No available model worked. Set GOOGLE_LLM_MODEL to a supported model, "
                "for example 'gemma-3-4b-it' or 'gemma-3-12b-it'."
            ) from err

    def _stream_with_fallback(self, prompt: str) -> Iterator[str]:
        def stream_from_current_llm() -> Iterator[str]:
            for chunk in self.llm.stream([prompt]):
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

        try:
            yield from stream_from_current_llm()
            return
        except Exception as err:
            text = str(err)
            if not self._should_fallback_for_error(text):
                raise

            candidates = [m for m in self.FALLBACK_MODELS if m != self.llm_model]
            print(f"[WARN] Model '{self.llm_model}' stream failed. Trying fallbacks: {candidates}")
            for model in candidates:
                try:
                    self._try_switch_model(model)
                    yield from stream_from_current_llm()
                    return
                except Exception as fallback_err:
                    print(f"[WARN] Fallback model '{model}' stream failed: {fallback_err}")

            raise RuntimeError(
                "No available model worked for streaming. Set GOOGLE_LLM_MODEL to a supported model."
            ) from err

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
        return re.findall(r"[a-zA-Z0-9_\-]+", str(text or "").lower())

    def _ensure_bm25_index(self) -> bool:
        if self._bm25_index is not None:
            return True
        if BM25Okapi is None:
            return False

        rows = list(self.vectorstore.metadata or [])
        if not rows:
            return False

        tokenized = []
        mapped_rows = []
        for row_index, row in enumerate(rows):
            metadata = dict(row or {})
            text = str(metadata.get("text", "") or "")
            tokens = self._tokenize_for_bm25(text)
            if not tokens:
                continue
            tokenized.append(tokens)
            mapped_rows.append((row_index, metadata))

        if not tokenized:
            return False

        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_tokens = tokenized
        self._bm25_rows = mapped_rows
        print(f"[INFO] BM25 index initialized with {len(mapped_rows)} chunks")
        return True

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict]:
        if not self._ensure_bm25_index():
            return []

        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

        results = []
        for rank_pos, internal_idx in enumerate(ranked_indices[:top_k], start=1):
            row_index, metadata = self._bm25_rows[internal_idx]
            results.append(
                {
                    "index": row_index,
                    "distance": float("inf"),
                    "metadata": metadata,
                    "bm25_score": float(scores[internal_idx]),
                    "bm25_rank": rank_pos,
                }
            )
        return results

    @staticmethod
    def _rrf_fuse(dense_results: list[dict], sparse_results: list[dict], dense_weight: float = 0.6, sparse_weight: float = 0.4, rrf_k: int = 60) -> list[dict]:
        merged: dict[int, dict] = {}

        for rank, row in enumerate(dense_results, start=1):
            key = int(row.get("index", -1))
            if key < 0:
                continue
            payload = dict(row)
            payload["dense_rank"] = rank
            payload["sparse_rank"] = None
            payload["hybrid_score"] = dense_weight * (1.0 / (rrf_k + rank))
            merged[key] = payload

        for rank, row in enumerate(sparse_results, start=1):
            key = int(row.get("index", -1))
            if key < 0:
                continue
            if key not in merged:
                payload = dict(row)
                payload["dense_rank"] = None
                payload["distance"] = float(row.get("distance", float("inf")))
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
        value = text or ""
        for old, new in replacements.items():
            value = value.replace(old, new)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

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
            distance = float(r.get("distance", 0.0))
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
            source = str(metadata.get("source", "Unknown source") or "Unknown source")
            preview = text[:150] + ("..." if len(text) > 150 else "")
            source_type = self._source_type_from_value(source, metadata)

            sources.append(
                {
                    "index": idx,
                    "source": source,
                    "type": source_type,
                    "preview": preview,
                    "text": text,
                    "score": float(row.get("score", 0.0)),
                    "distance": float(row.get("distance", 0.0)),
                    "file_id": metadata.get("file_id"),
                    "page": metadata.get("page"),
                    "title": metadata.get("title", ""),
                }
            )
        return sources

    def _build_answer_prompt(
        self,
        query: str,
        sources: list[dict],
        direct_snippets: list[str],
        memory_context: str | None,
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

        return (
            "Answer the user query using only relevant information from the retrieved context.\n"
            "Whenever you make a factual claim from context, add inline citations like [1], [2].\n"
            "If multiple sources support one claim, cite all relevant references.\n"
            "Do not fabricate citations.\n\n"
            f"{memory_block}"
            f"{direct_match_block}"
            f"User query: {query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "If context is insufficient, say what is missing briefly."
        )

    def _retrieve_for_answer(self, query: str, top_k: int, memory_context: str | None = None) -> tuple[list[dict], str]:
        query_terms = self._query_terms(query)
        strong_terms = [t for t in query_terms if len(t) >= 7]
        retrieval_hint = self._memory_retrieval_hint(memory_context)
        use_memory_for_retrieval = bool(retrieval_hint) and len(strong_terms) == 0
        retrieval_query = query if not use_memory_for_retrieval else f"{query} {retrieval_hint}"

        candidate_k = max(top_k * 20, 120)
        dense_results = self.vectorstore.query(retrieval_query, top_k=candidate_k)
        sparse_results = self._bm25_retrieve(retrieval_query, top_k=candidate_k)
        fused_results = self._rrf_fuse(dense_results, sparse_results)
        ranked = self._rank_results(retrieval_query, fused_results)
        parent_expanded = self._expand_to_parent_context(ranked, top_k=top_k)
        return parent_expanded, retrieval_query

    def answer_with_sources(self, query: str, top_k: int = 5, memory_context: str | None = None) -> dict:
        ranked_results, _ = self._retrieve_for_answer(query, top_k, memory_context)
        sources = self._build_source_payload(ranked_results)
        if not sources:
            return {"answer": "No relevant documents found.", "sources": []}

        direct_snippets = self._extract_direct_snippets(query, ranked_results)
        prompt = self._build_answer_prompt(query, sources, direct_snippets, memory_context)
        response = self._invoke_with_fallback(prompt)
        answer = str(response.content or "").strip() or "No relevant documents found."
        return {"answer": answer, "sources": sources}

    def stream_answer_with_sources(self, query: str, top_k: int = 5, memory_context: str | None = None) -> tuple[list[dict], Iterator[str]]:
        ranked_results, _ = self._retrieve_for_answer(query, top_k, memory_context)
        sources = self._build_source_payload(ranked_results)
        if not sources:
            return [], iter(["No relevant documents found."])

        direct_snippets = self._extract_direct_snippets(query, ranked_results)
        prompt = self._build_answer_prompt(query, sources, direct_snippets, memory_context)
        return sources, self._stream_with_fallback(prompt)

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