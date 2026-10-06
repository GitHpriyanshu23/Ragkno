import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.embedding import EmbeddingPipeline


class ChromaVectorStore:
    """Tenant-scoped Chroma storage with deterministic source and chunk IDs."""

    def __init__(self, persist_dir: str = "chroma_store", collection_name: str = "ragkno_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200, load_model: bool = True):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_device = os.getenv("RAG_EMBEDDING_DEVICE", "cpu").strip().lower() or "cpu"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model, device=self.embedding_device) if load_model else None
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def canonical_source(source: str, source_type: str = "upload", file_id: str | None = None) -> str:
        value = str(source or "").strip()
        if source_type == "drive" and file_id:
            return f"drive:{file_id}"
        if source_type == "url":
            parts = urlsplit(value)
            host = (parts.hostname or "").lower()
            if parts.port:
                host = f"{host}:{parts.port}"
            return urlunsplit((parts.scheme.lower(), host, parts.path or "/", parts.query, ""))
        return value

    @classmethod
    def source_id_for(cls, user_id: str, source: str, source_type: str = "upload", file_id: str | None = None) -> str:
        canonical = cls.canonical_source(source, source_type, file_id)
        digest = hashlib.sha256(f"{user_id}\0{source_type}\0{canonical}".encode("utf-8")).hexdigest()
        return f"src_{digest[:32]}"

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        records = self.collection.get(include=["metadatas", "documents"])
        results = []
        for cid, doc, meta in zip(records.get("ids") or [], records.get("documents") or [], records.get("metadatas") or []):
            item = dict(meta or {})
            item["text"] = doc
            item["chunk_id"] = cid
            results.append(item)
        return results

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in (meta or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, tuple)):
                clean[key] = ", ".join(str(item) for item in value)
            else:
                clean[key] = str(value)
        return clean

    @staticmethod
    def _tenant_where(user_id: str, source_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        uid = str(user_id or "").strip()
        if not uid or uid in {"*", "all"}:
            raise ValueError("A concrete user_id is required for vector-store access")
        if source_ids:
            return {"$and": [{"user_id": uid}, {"source_id": {"$in": list(dict.fromkeys(source_ids))}}]}
        return {"user_id": uid}

    def get_user_metadata(self, user_id: str, source_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        records = self.collection.get(where=self._tenant_where(user_id, source_ids), include=["metadatas", "documents"])
        rows = []
        for cid, doc, meta in zip(records.get("ids") or [], records.get("documents") or [], records.get("metadatas") or []):
            row = dict(meta or {})
            row["text"] = doc
            row["chunk_id"] = cid
            rows.append(row)
        return rows

    def build_from_documents(self, documents: List[Any], user_id: str):
        return self.add_documents(documents, user_id=user_id)

    def add_documents(self, documents: List[Any], user_id: str, progress=None) -> List[Dict[str, Any]]:
        uid = str(user_id or "").strip()
        if not uid or uid in {"*", "all", "system", "guest"}:
            raise ValueError("Authenticated user_id is required for ingestion")
        if not documents:
            return []
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")

        pipeline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, model=self.model)
        if progress:
            progress("Splitting document text", 10)
        chunks = pipeline.chunk_documents(documents)
        if not chunks:
            return []
        embeddings = pipeline.embed_chunks(chunks, progress=progress)
        by_source: Dict[str, Dict[str, Any]] = {}
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        texts: List[str] = []

        for index, chunk in enumerate(chunks):
            raw = dict(getattr(chunk, "metadata", {}) or {})
            source = str(raw.get("source") or "unknown")
            source_type = str(raw.get("source_type") or "upload").lower()
            file_id = str(raw.get("file_id") or "") or None
            source_id = str(raw.get("source_id") or self.source_id_for(uid, source, source_type, file_id))
            canonical = self.canonical_source(source, source_type, file_id)
            content = str(chunk.page_content or "").strip()
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            position = f"{raw.get('page', 'na')}:{raw.get('parent_index', 0)}:{raw.get('child_index', index)}"
            chunk_id = f"chk_{hashlib.sha256(f'{uid}:{source_id}:{position}:{content_hash}'.encode()).hexdigest()}"
            raw.update({"user_id": uid, "source_id": source_id, "source": source, "source_key": canonical, "source_type": source_type, "content_hash": content_hash, "title": raw.get("title") or raw.get("name") or Path(source).name or source})
            ids.append(chunk_id)
            metadatas.append(self._sanitize_metadata(raw))
            texts.append(content)
            summary = by_source.setdefault(source_id, {"source_id": source_id, "source_key": canonical, "source_name": raw["title"], "source_type": source_type, "chunk_count": 0, "pages": set()})
            summary["chunk_count"] += 1
            summary["pages"].add(str(raw.get("page", "0")))

        old_by_source: Dict[str, List[str]] = {}
        for source_id in by_source:
            existing = self.collection.get(where={"$and": [{"user_id": uid}, {"source_id": source_id}]}, include=[])
            old_by_source[source_id] = list(existing.get("ids") or [])
        for start in range(0, len(ids), 500):
            end = min(start + 500, len(ids))
            self.collection.upsert(ids=ids[start:end], documents=texts[start:end], metadatas=metadatas[start:end], embeddings=embeddings[start:end].tolist())
            if progress:
                progress("Saving searchable chunks", 80 + 18 * end / len(ids))
        new_ids = set(ids)
        stale = [cid for values in old_by_source.values() for cid in values if cid not in new_ids]
        if stale:
            self.collection.delete(ids=stale)

        results = []
        for item in by_source.values():
            item["page_count"] = len(item.pop("pages"))
            results.append(item)
        return results

    def query(self, query_text: str, top_k: int, user_id: str, source_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")
        try:
            embedding = self.model.encode([query_text], device=self.embedding_device)
        except RuntimeError as exc:
            detail = str(exc).lower()
            accelerator_oom = "out of memory" in detail and any(name in detail for name in ("mps", "cuda"))
            if not accelerator_oom or self.embedding_device == "cpu":
                raise
            print(f"[WARN] {self.embedding_device.upper()} embedding memory exhausted; retrying on CPU")
            self.embedding_device = "cpu"
            self.model.to("cpu")
            embedding = self.model.encode([query_text], device="cpu")
        return self.search(embedding.tolist(), top_k, user_id, source_ids)

    def search(self, query_embedding: Any, top_k: int, user_id: str, source_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        emb_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else list(query_embedding)
        if emb_list and isinstance(emb_list[0], (int, float)):
            emb_list = [emb_list]
        try:
            result = self.collection.query(query_embeddings=emb_list, n_results=max(1, min(int(top_k), 400)), include=["documents", "metadatas", "distances"], where=self._tenant_where(user_id, source_ids))
        except Exception as exc:
            raise RuntimeError("Tenant-scoped vector query failed") from exc
        rows = []
        if not result or not result.get("ids") or not result["ids"][0]:
            return rows
        for rank, (cid, distance, document, meta) in enumerate(zip(result["ids"][0], (result.get("distances") or [[]])[0], (result.get("documents") or [[]])[0], (result.get("metadatas") or [[]])[0])):
            full_meta = dict(meta or {})
            if full_meta.get("user_id") != user_id:
                continue
            full_meta["text"] = document
            rows.append({"index": rank, "id": cid, "distance": float(distance), "metadata": full_meta})
        return rows

    def remove_source(self, source_key: str, user_id: str) -> int:
        uid, key = str(user_id or "").strip(), str(source_key or "").strip()
        if not uid or not key:
            raise ValueError("user_id and source identifier are required")
        records = self.collection.get(where={"user_id": uid}, include=["metadatas"])
        matching = []
        for cid, meta in zip(records.get("ids") or [], records.get("metadatas") or []):
            meta = meta or {}
            if key in {str(meta.get("source_id") or ""), str(meta.get("source_key") or ""), str(meta.get("source") or "")}:
                matching.append(cid)
        if matching:
            self.collection.delete(ids=matching)
        return len(matching)

    def get_user_sources(self, user_id: str) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for meta in self.get_user_metadata(user_id):
            source_id = str(meta.get("source_id") or "")
            if not source_id:
                continue
            item = grouped.setdefault(source_id, {"source_id": source_id, "source_key": meta.get("source_key") or meta.get("source"), "source": meta.get("source"), "title": meta.get("title") or meta.get("source"), "source_type": meta.get("source_type") or "upload", "user_id": user_id, "chunk_count": 0})
            item["chunk_count"] += 1
        return list(grouped.values())

    def purge_system_documents(self) -> int:
        records = self.collection.get(where={"user_id": "system"}, include=[])
        ids = records.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def migrate_legacy_records(self) -> Dict[str, int]:
        """Assign deterministic IDs to legacy tenant chunks without re-embedding them."""
        records = self.collection.get(include=["metadatas", "documents", "embeddings"])
        migrated = 0
        skipped = 0
        new_ids, new_docs, new_meta, new_embeddings, old_ids = [], [], [], [], []
        embeddings = records.get("embeddings")
        if embeddings is None:
            embeddings = []
        for cid, document, meta, embedding in zip(records.get("ids") or [], records.get("documents") or [], records.get("metadatas") or [], embeddings):
            metadata = dict(meta or {})
            if metadata.get("source_id"):
                skipped += 1
                continue
            user_id = str(metadata.get("user_id") or "")
            if not user_id or user_id in {"system", "guest"}:
                skipped += 1
                continue
            source = str(metadata.get("source_raw") or metadata.get("source") or "unknown")
            source_type = str(metadata.get("source_type") or ("url" if source.startswith(("http://", "https://")) else "upload"))
            if source_type == "file":
                source_type = "upload"
            file_id = str(metadata.get("file_id") or "") or None
            source_id = self.source_id_for(user_id, source, source_type, file_id)
            content_hash = hashlib.sha256(str(document or "").encode("utf-8")).hexdigest()
            new_id = f"chk_{hashlib.sha256(f'{user_id}:{source_id}:{cid}:{content_hash}'.encode()).hexdigest()}"
            metadata.update({"source_id": source_id, "source": source, "source_key": self.canonical_source(source, source_type, file_id), "source_type": source_type, "content_hash": content_hash})
            new_ids.append(new_id)
            new_docs.append(document)
            new_meta.append(self._sanitize_metadata(metadata))
            new_embeddings.append(embedding)
            old_ids.append(cid)
            migrated += 1
        if new_ids:
            self.collection.upsert(ids=new_ids, documents=new_docs, metadatas=new_meta, embeddings=new_embeddings)
            self.collection.delete(ids=old_ids)
        return {"migrated": migrated, "skipped": skipped}
