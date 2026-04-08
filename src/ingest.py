import tempfile
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from requests.exceptions import SSLError


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _safe_suffix(name: str) -> str:
    suffix = Path(name or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
    return suffix


def load_uploaded_file(file_name: str, content: bytes) -> List[Document]:
    suffix = _safe_suffix(file_name)
    if not content:
        return []
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File '{file_name}' exceeds 50MB limit")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(str(tmp_path)).load()
        elif suffix == ".txt":
            docs = TextLoader(str(tmp_path), encoding="utf-8").load()
        else:
            docs = Docx2txtLoader(str(tmp_path)).load()

        for doc in docs:
            meta = dict(doc.metadata or {})
            meta["source"] = file_name
            meta["source_type"] = "upload"
            doc.metadata = meta
        return docs
    finally:
        tmp_path.unlink(missing_ok=True)


def load_uploaded_files(files: Iterable[tuple[str, bytes]]) -> List[Document]:
    documents: List[Document] = []
    for file_name, content in files:
        documents.extend(load_uploaded_file(file_name=file_name, content=content))
    return documents


def validate_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported")
    if not parsed.netloc:
        raise ValueError("Invalid URL")
    return parsed.geturl()


def load_url_content(url: str, timeout: int = 15) -> List[Document]:
    clean_url = validate_url(url)
    try:
        response = requests.get(clean_url, timeout=timeout, headers={"User-Agent": "RAGKNOBot/1.0"})
    except SSLError:
        response = requests.get(clean_url, timeout=timeout, headers={"User-Agent": "RAGKNOBot/1.0"}, verify=False)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.extract()

    title = (soup.title.string or "").strip() if soup.title else ""
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    if not text:
        raise ValueError("No readable text found on the provided URL")

    content = f"Title: {title}\n\n{text}" if title else text
    return [
        Document(
            page_content=content,
            metadata={
                "source": clean_url,
                "source_type": "url",
                "title": title,
            },
        )
    ]
