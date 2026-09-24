import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        collection_name: str = "ragkno_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[INFO] ChromaVectorStore initialized at '{self.persist_dir}' [collection='{self.collection_name}']")

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Compatibility property for components that read raw metadata list."""
        try:
            records = self.collection.get(include=["metadatas", "documents"])
            results = []
            documents = records.get("documents") or []
            metadatas = records.get("metadatas") or []
            for doc, meta in zip(documents, metadatas):
                item = dict(meta or {})
                item["text"] = doc
                results.append(item)
            return results
        except Exception as e:
            print(f"[WARN] Error fetching metadata from Chroma: {e}")
            return []

    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB metadata requires primitive types (str, int, float, bool)."""
        clean = {}
        for k, v in (meta or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                clean[k] = ", ".join(str(item) for item in v)
            elif isinstance(v, dict):
                clean[k] = str(v)
            else:
                clean[k] = str(v)
        return clean

    def _normalize_source_key(self, source: str) -> str:
        return str(source or "").strip().lower().rstrip("/")

    def build_from_documents(self, documents: List[Any], user_id: str = "system"):
        print(f"[INFO] Building Chroma vector store from {len(documents)} raw documents (user_id={user_id})...")
        self.add_documents(documents, user_id=user_id)

    def add_documents(self, documents: List[Any], user_id: str = "system"):
        if not documents:
            return

        user_key = str(user_id or "system").strip()
        print(f"[INFO] Appending {len(documents)} raw documents to Chroma store for user '{user_key}'...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        if not chunks:
            print("[WARN] No chunks generated from documents.")
            return

        texts = [chunk.page_content for chunk in chunks]
        embeddings = emb_pipe.embed_chunks(chunks)

        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        docs: List[str] = []

        for idx, chunk in enumerate(chunks):
            raw_meta = dict(getattr(chunk, "metadata", {}) or {})
            source = str(raw_meta.get("source", "unknown"))
            normalized_source = self._normalize_source_key(source)

            raw_meta["user_id"] = user_key
            raw_meta["source"] = normalized_source
            raw_meta["source_raw"] = source
            if "title" not in raw_meta:
                raw_meta["title"] = raw_meta.get("name") or Path(source).name or normalized_source

            clean_meta = self._sanitize_metadata(raw_meta)

            chunk_id = f"{user_key}::{normalized_source}::{idx}::{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            metadatas.append(clean_meta)
            docs.append(chunk.page_content)

        # Batch upsert into ChromaDB in chunks of 500
        batch_size = 500
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.upsert(
                ids=ids[i:end],
                documents=docs[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end].tolist(),
            )

        print(f"[INFO] Successfully added {total} chunks to Chroma collection '{self.collection_name}'.")

    def _build_where_clause(self, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not user_id or user_id.strip() in ("*", "all"):
            return None
        uid = str(user_id).strip()
        # Strictly isolate user documents - only documents belonging to this user
        return {"user_id": uid}

    def purge_system_documents(self) -> int:
        """Purge all legacy system documents from the collection."""
        try:
            records = self.collection.get(where={"user_id": "system"})
            ids = records.get("ids") or []
            if ids:
                self.collection.delete(where={"user_id": "system"})
                print(f"[INFO] Purged {len(ids)} system-tagged chunks from Chroma.")
                return len(ids)
        except Exception as e:
            print(f"[WARN] Failed to purge system chunks: {e}")
        return 0

    def query(self, query_text: str, top_k: int = 5, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        print(f"[INFO] Chroma query: '{query_text[:60]}...' (user_id={user_id}, top_k={top_k})")
        query_emb = self.model.encode([query_text]).tolist()
        return self.search(query_emb, top_k=top_k, user_id=user_id)

    def search(self, query_embedding: Any, top_k: int = 5, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if isinstance(query_embedding, np.ndarray):
            emb_list = query_embedding.tolist()
        else:
            emb_list = list(query_embedding)

        if emb_list and isinstance(emb_list[0], (int, float)):
            emb_list = [emb_list]

        where_clause = self._build_where_clause(user_id)

        try:
            kwargs: Dict[str, Any] = {
                "query_embeddings": emb_list,
                "n_results": max(1, top_k),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_clause:
                kwargs["where"] = where_clause

            res = self.collection.query(**kwargs)
        except Exception as e:
            # Fallback without where filter if where fails (e.g. empty collection)
            print(f"[WARN] Chroma query with filter failed: {e}. Retrying without filter...")
            try:
                res = self.collection.query(
                    query_embeddings=emb_list,
                    n_results=max(1, top_k),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e2:
                print(f"[ERROR] Chroma query failed: {e2}")
                return []

        results: List[Dict[str, Any]] = []
        if not res or not res.get("ids") or not res["ids"][0]:
            return results

        ids = res["ids"][0]
        distances = res.get("distances", [[]])[0]
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]

        for idx, (cid, dist, doc, meta) in enumerate(zip(ids, distances, documents, metadatas)):
            full_meta = dict(meta or {})
            full_meta["text"] = doc
            results.append({
                "index": idx,
                "id": cid,
                "distance": dist,
                "metadata": full_meta,
            })

        return results

    def remove_source(self, source_key: str, user_id: Optional[str] = None) -> int:
        normalized_key = self._normalize_source_key(source_key)
        if not normalized_key:
            return 0

        where_clause: Dict[str, Any]
        if user_id and user_id.strip() not in ("*", "all"):
            uid = str(user_id).strip()
            where_clause = {"$and": [{"user_id": uid}, {"source": normalized_key}]}
        else:
            where_clause = {"source": normalized_key}

        try:
            records = self.collection.get(where=where_clause)
            count = len(records.get("ids") or [])
            if count > 0:
                self.collection.delete(where=where_clause)
                print(f"[INFO] Removed {count} chunk(s) from Chroma for source '{normalized_key}'.")
            return count
        except Exception as e:
            print(f"[ERROR] Failed to delete source from Chroma: {e}")
            return 0

    def get_user_sources(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        where_clause = self._build_where_clause(user_id)
        try:
            kwargs = {"include": ["metadatas"]}
            if where_clause:
                kwargs["where"] = where_clause
            records = self.collection.get(**kwargs)
            metadatas = records.get("metadatas") or []
        except Exception as e:
            print(f"[WARN] Failed to list user sources from Chroma: {e}")
            return []

        grouped: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            src = meta.get("source", "")
            if not src:
                continue
            if src not in grouped:
                grouped[src] = {
                    "source": src,
                    "title": meta.get("title") or meta.get("source_raw") or src,
                    "source_type": meta.get("source_type") or "file",
                    "user_id": meta.get("user_id", "system"),
                    "chunk_count": 0,
                }
            grouped[src]["chunk_count"] += 1

        return list(grouped.values())
