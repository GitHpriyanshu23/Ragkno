"""
src/drive_loader.py

Handles Google Drive OAuth2 flow and document ingestion.
Supports PDF and TXT files from Google Drive.
"""

import io
import os
import json
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_core.documents import Document

# Always load .env from the project root, regardless of CWD
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ----- Config ----------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

TOKEN_PATH = Path(__file__).parent.parent / "token.json"

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    # Google Docs → export as plain text
    "application/vnd.google-apps.document": "gdoc",
}

# In-memory cache for OAuth PKCE verifier keyed by OAuth state.
# This is sufficient for local dev in a single backend process.
_OAUTH_CODE_VERIFIERS: dict[str, tuple[str, float]] = {}
_OAUTH_VERIFIER_TTL_SECONDS = 600


def _cleanup_oauth_cache() -> None:
    now = time.time()
    expired_states = [
        state for state, (_, ts) in _OAUTH_CODE_VERIFIERS.items()
        if now - ts > _OAUTH_VERIFIER_TTL_SECONDS
    ]
    for state in expired_states:
        _OAUTH_CODE_VERIFIERS.pop(state, None)

# ----- OAuth Helpers ---------------------------------------------------------


def _build_flow() -> Flow:
    client_config = {
        "web": {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback"),
    )
    return flow


def get_auth_url() -> str:
    """Return the Google OAuth consent URL and cache PKCE verifier."""
    _cleanup_oauth_cache()
    flow = _build_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    # google-auth-oauthlib stores this after authorization_url() call.
    if not flow.code_verifier:
        raise RuntimeError("Failed to initialize OAuth code verifier.")
    _OAUTH_CODE_VERIFIERS[state] = (flow.code_verifier, time.time())

    return auth_url


def exchange_code(code: str, state: str | None = None) -> Credentials:
    """Exchange an auth code for credentials and persist token.json."""
    _cleanup_oauth_cache()
    flow = _build_flow()

    if state:
        verifier_entry = _OAUTH_CODE_VERIFIERS.pop(state, None)
        if verifier_entry:
            flow.code_verifier = verifier_entry[0]

    flow.fetch_token(code=code)
    creds = flow.credentials
    _save_token(creds)
    return creds


def _save_token(creds: Credentials):
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
    }
    TOKEN_PATH.write_text(json.dumps(token_data))
    print(f"[INFO] Token saved to {TOKEN_PATH}")


def load_credentials() -> Credentials | None:
    """Load and (if needed) refresh credentials from token.json."""
    if not TOKEN_PATH.exists():
        return None
    raw = json.loads(TOKEN_PATH.read_text())
    creds = Credentials(
        token=raw["token"],
        refresh_token=raw.get("refresh_token"),
        token_uri=raw.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=raw["client_id"],
        client_secret=raw["client_secret"],
        scopes=raw.get("scopes", SCOPES),
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)
        print("[INFO] Token refreshed.")
    return creds


def is_connected() -> bool:
    return TOKEN_PATH.exists()


def disconnect():
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
        print("[INFO] token.json deleted — disconnected from Drive.")


# ----- Drive API helpers -----------------------------------------------------


def _list_drive_files_with_service(service) -> List[dict]:
    """Internal: list files using an already-built Drive service object."""
    mime_query = " or ".join(
        [f"mimeType='{m}'" for m in SUPPORTED_MIME_TYPES]
    )
    query = f"({mime_query}) and trashed=false"
    results = []
    page_token = None

    while True:
        resp = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                pageSize=100,
            )
            .execute()
        )
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    print(f"[INFO] Found {len(results)} supported files in Drive.")
    return results


def list_drive_files(creds: Credentials) -> List[dict]:
    """Public: build a Drive service and list supported files."""
    service = build("drive", "v3", credentials=creds)
    return _list_drive_files_with_service(service)


def _download_file(service, file_id: str, mime_type: str) -> bytes:
    """Download raw file bytes; for Google Docs export as plain text."""
    if mime_type == "application/vnd.google-apps.document":
        req = service.files().export_media(fileId=file_id, mimeType="text/plain")
    else:
        req = service.files().get_media(fileId=file_id)

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read()


# ----- LangChain Document Builder -------------------------------------------


def load_documents_from_drive(creds: Credentials, file_ids: List[str] | None = None) -> List[Document]:
    """
    Connect to Drive, download all supported files, and return
    a list of LangChain Document objects ready for embedding.
    """
    # Build the service once and reuse it for both listing and downloading
    service = build("drive", "v3", credentials=creds)
    files = _list_drive_files_with_service(service)

    if file_ids is not None:
        selected_ids = set(file_ids)
        files = [f for f in files if f["id"] in selected_ids]
        print(f"[INFO] Sync limited to {len(files)} selected files.")

    documents: List[Document] = []

    for f in files:
        file_id = f["id"]
        name = f["name"]
        mime = f["mimeType"]
        print(f"[INFO] Processing: {name} ({mime})")

        try:
            raw_bytes = _download_file(service, file_id, mime)

            if mime == "application/pdf":
                # Use pypdf to extract text from PDF bytes
                import pypdf

                reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": f"drive://{name}",
                                    "file_id": file_id,
                                    "page": page_num,
                                },
                            )
                        )
            else:
                # TXT or exported Google Doc
                text = raw_bytes.decode("utf-8", errors="replace")
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": f"drive://{name}",
                                "file_id": file_id,
                            },
                        )
                    )
        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {e}")

    print(f"[INFO] Loaded {len(documents)} documents from Drive.")
    return documents
