import re
from typing import List, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        parent_chunk_size: int = 2200,
        parent_chunk_overlap: int = 250,
        semantic_threshold: float = 0.55,
        semantic_min_chars: int = 350,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.semantic_min_chars = semantic_min_chars
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model : {model_name}")

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return [part.strip() for part in parts if part and part.strip()]

    def _semantic_child_chunks(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        sentence_embeddings = self.model.encode(sentences)
        chunks: list[str] = []
        current_sentences = [sentences[0]]
        current_len = len(sentences[0])

        for idx in range(1, len(sentences)):
            prev_embedding = sentence_embeddings[idx - 1]
            curr_embedding = sentence_embeddings[idx]
            similarity = float(np.dot(prev_embedding, curr_embedding) / (np.linalg.norm(prev_embedding) * np.linalg.norm(curr_embedding) + 1e-12))

            sentence = sentences[idx]
            projected_len = current_len + 1 + len(sentence)
            boundary_by_size = projected_len > self.chunk_size
            boundary_by_semantics = similarity < self.semantic_threshold and current_len >= self.semantic_min_chars

            if boundary_by_size or boundary_by_semantics:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = [sentence]
                current_len = len(sentence)
            else:
                current_sentences.append(sentence)
                current_len = projected_len

        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

        # Keep overly large chunks bounded with overlap fallback.
        bounded_chunks: list[str] = []
        overflow_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                bounded_chunks.append(chunk)
                continue
            bounded_chunks.extend(overflow_splitter.split_text(chunk))
        return [chunk for chunk in bounded_chunks if chunk.strip()]

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks: list[Document] = []
        for doc_index, doc in enumerate(documents):
            parent_texts = parent_splitter.split_text(str(getattr(doc, "page_content", "") or ""))
            if not parent_texts:
                continue

            base_metadata = dict(getattr(doc, "metadata", {}) or {})
            source = str(base_metadata.get("source", "unknown"))
            page = base_metadata.get("page", "na")

            for parent_index, parent_text in enumerate(parent_texts):
                parent_id = f"{source}::p{page}::d{doc_index}::c{parent_index}"
                child_texts = self._semantic_child_chunks(parent_text)
                for child_index, child_text in enumerate(child_texts):
                    metadata = dict(base_metadata)
                    metadata.update(
                        {
                            "parent_id": parent_id,
                            "parent_index": parent_index,
                            "child_index": child_index,
                            "parent_text": parent_text,
                        }
                    )
                    chunks.append(Document(page_content=child_text, metadata=metadata))

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] GEnerating embeddings for {len(texts)} chunks")
        embeddings = self.model.encode(texts, show_progress_bar = True)
        print(f"[INFO] Embeddigns shape : {embeddings.shape}")
        return embeddings