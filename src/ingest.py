import tempfile
import ipaddress
import socket
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_URL_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5


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

        docs = [doc for doc in docs if str(getattr(doc, "page_content", "") or "").strip()]
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
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL hostname is required")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError("URL hostname could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Private, loopback, link-local, and reserved addresses are not allowed")
    return parsed.geturl()


def load_url_content(url: str, timeout: int = 15) -> List[Document]:
    clean_url = validate_url(url)
    response = None
    session = requests.Session()
    for _ in range(MAX_REDIRECTS + 1):
        response = session.get(clean_url, timeout=timeout, headers={"User-Agent": "RAGKNOBot/1.0"}, allow_redirects=False, stream=True)
        if response.is_redirect or response.is_permanent_redirect:
            target = response.headers.get("location")
            response.close()
            if not target:
                raise ValueError("Redirect response did not include a destination")
            clean_url = validate_url(urljoin(clean_url, target))
            continue
        break
    else:
        raise ValueError("Too many redirects")
    if response is None:
        raise ValueError("URL could not be fetched")
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
        response.close()
        raise ValueError("URL must return HTML or plain text")
    chunks = []
    total = 0
    for block in response.iter_content(chunk_size=64 * 1024):
        total += len(block)
        if total > MAX_URL_BYTES:
            response.close()
            raise ValueError("URL response exceeds 5MB limit")
        chunks.append(block)
    encoding = response.encoding or "utf-8"
    body = b"".join(chunks).decode(encoding, errors="replace")
    response.close()

    soup = BeautifulSoup(body, "html.parser")
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
