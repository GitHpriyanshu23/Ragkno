import os
import json
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        # Preserve upstream metadata (e.g., source filename/page) so query-time
        # ranking can prefer relevant files like resumes over noisy chat logs.
        metadatas = []
        for chunk in chunks:
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            metadata["text"] = chunk.page_content
            metadatas.append(metadata)

        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_documents(self, documents: List[Any]):
        if not documents:
            return

        print(f"[INFO] Appending {len(documents)} raw documents to vector store...")
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        meta_json_path = os.path.join(self.persist_dir, "metadata.json")

        if os.path.exists(faiss_path) and (os.path.exists(meta_path) or os.path.exists(meta_json_path)):
            self.load()

        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        metadatas = []
        for chunk in chunks:
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            metadata["text"] = chunk.page_content
            metadatas.append(metadata)

        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Appended and saved chunks to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        meta_json_path = os.path.join(self.persist_dir, "metadata.json")
        temp_meta_path = f"{meta_path}.tmp"
        temp_meta_json_path = f"{meta_json_path}.tmp"

        faiss.write_index(self.index, faiss_path)
        with open(temp_meta_path, "wb") as f:
            pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_meta_path, meta_path)

        with open(temp_meta_json_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        os.replace(temp_meta_json_path, meta_json_path)

        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        meta_json_path = os.path.join(self.persist_dir, "metadata.json")
        self.index = faiss.read_index(faiss_path)

        try:
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        except Exception as pickle_error:
            if not os.path.exists(meta_json_path):
                raise RuntimeError(
                    f"Failed to load vector metadata from {meta_path}: {pickle_error}"
                ) from pickle_error

            with open(meta_json_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"[WARN] Loaded metadata from JSON backup because pickle load failed: {pickle_error}")

        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

    def _normalize_source_key(self, source: str) -> str:
        return str(source or "").strip().lower().rstrip("/")

    def remove_source(self, source_key: str) -> int:
        """Remove all chunks belonging to a source and rebuild the index."""
        normalized_key = self._normalize_source_key(source_key)
        if not normalized_key:
            return 0

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        meta_json_path = os.path.join(self.persist_dir, "metadata.json")
        if not (os.path.exists(faiss_path) and (os.path.exists(meta_path) or os.path.exists(meta_json_path))):
            return 0

        self.load()
        before_count = len(self.metadata)
        kept_rows = []
        kept_texts = []

        for row in self.metadata:
            metadata = dict(row or {})
            source = self._normalize_source_key(metadata.get("source", ""))
            if source == normalized_key:
                continue

            text = str(metadata.get("text", "") or "").strip()
            if not text:
                continue

            kept_rows.append(metadata)
            kept_texts.append(text)

        removed_count = before_count - len(kept_rows)
        if removed_count <= 0:
            return 0

        if not kept_rows:
            self.index = None
            self.metadata = []
            for path in (faiss_path, meta_path, meta_json_path):
                if os.path.exists(path):
                    os.remove(path)
            return removed_count

        embeddings = self.model.encode(kept_texts).astype('float32')
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata = kept_rows
        self.save()
        return removed_count

#Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("Priyanshu Urmaliya resume", top_k=3))