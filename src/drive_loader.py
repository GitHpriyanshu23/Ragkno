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

# Allow OAuth over HTTP on localhost during development
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
# Allow Google to return all previously granted user scopes without ScopeChangedError
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

# ----- Config ----------------------------------------------------------------

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive.readonly",
]

TOKEN_PATH = Path(__file__).parent.parent / "token.json"

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    # Google Docs → export as plain text
    "application/vnd.google-apps.document": "gdoc",
}

# Persistent cache for OAuth PKCE verifier keyed by OAuth state.
_VERIFIERS_FILE = _PROJECT_ROOT / "data" / "oauth_verifiers.json"
_OAUTH_CODE_VERIFIERS: dict[str, tuple[str, float, str | None]] = {}
_OAUTH_VERIFIER_TTL_SECONDS = 1800


def save_oauth_verifier(state: str, verifier: str, user_id: str | None = None) -> None:
    now = time.time()
    _OAUTH_CODE_VERIFIERS[state] = (verifier, now, user_id)
    try:
        _VERIFIERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if _VERIFIERS_FILE.exists():
            try:
                data = json.loads(_VERIFIERS_FILE.read_text())
            except Exception:
                data = {}
        data[state] = [verifier, now, user_id]
        _VERIFIERS_FILE.write_text(json.dumps(data))
    except Exception as e:
        print(f"[WARN] Failed to write OAuth verifier to disk: {e}")


def pop_oauth_verifier_with_user(state: str) -> tuple[str | None, str | None]:
    now = time.time()
    verifier = None
    user_id = None
    if state in _OAUTH_CODE_VERIFIERS:
        entry = _OAUTH_CODE_VERIFIERS.pop(state)
        verifier = entry[0]
        user_id = entry[2] if len(entry) > 2 else None

    try:
        if _VERIFIERS_FILE.exists():
            data = json.loads(_VERIFIERS_FILE.read_text())
            if state in data:
                entry = data.pop(state)
                verifier = verifier or entry[0]
                if not user_id and len(entry) > 2:
                    user_id = entry[2]
            # Clean up old entries
            data = {k: v for k, v in data.items() if now - v[1] <= _OAUTH_VERIFIER_TTL_SECONDS}
            _VERIFIERS_FILE.write_text(json.dumps(data))
    except Exception as e:
        print(f"[WARN] Failed to read/pop OAuth verifier from disk: {e}")

    return verifier, user_id


def pop_oauth_verifier(state: str) -> str | None:
    verifier, _ = pop_oauth_verifier_with_user(state)
    return verifier


_save_verifier = save_oauth_verifier
_pop_verifier = pop_oauth_verifier


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


def get_auth_url(user_id: str | None = None) -> str:
    """Return the Google OAuth consent URL and cache PKCE verifier linked to user_id."""
    flow = _build_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    # google-auth-oauthlib stores this after authorization_url() call.
    if not flow.code_verifier:
        raise RuntimeError("Failed to initialize OAuth code verifier.")
    save_oauth_verifier(state, flow.code_verifier, user_id=user_id)

    return auth_url


def exchange_code(code: str, state: str | None = None, user_id: str | None = None) -> Credentials:
    """Exchange an auth code for credentials and persist in Supabase for user_id."""
    flow = _build_flow()

    resolved_user_id = user_id
    if state:
        code_verifier, stored_uid = pop_oauth_verifier_with_user(state)
        if code_verifier:
            flow.code_verifier = code_verifier
        if not resolved_user_id and stored_uid:
            resolved_user_id = stored_uid

    flow.fetch_token(code=code)
    creds = flow.credentials

    if resolved_user_id:
        _save_user_token(resolved_user_id, creds)
    else:
        print("[WARN] exchange_code called without resolved user_id; token not persisted to DB")

    return creds


def _save_user_token(user_id: str, creds: Credentials) -> None:
    from src.database import Database
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "expiry_ts": creds.expiry.timestamp() if getattr(creds, "expiry", None) else None,
    }
    db = Database.get_instance()
    db.save_user_drive_token(user_id, token_data)
    print(f"[INFO] Drive token saved to Supabase for user {user_id}")


def load_credentials(user_id: str | None = None) -> Credentials | None:
    """Load and (if needed) refresh credentials for a specific user from Supabase."""
    uid = str(user_id or "").strip()
    if not uid:
        return None

    from src.database import Database
    db = Database.get_instance()
    raw = db.get_user_drive_token(uid)
    if not raw or not raw.get("token"):
        return None

    creds = Credentials(
        token=raw["token"],
        refresh_token=raw.get("refresh_token"),
        token_uri=raw.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=raw["client_id"],
        client_secret=raw["client_secret"],
        scopes=raw.get("scopes", SCOPES),
    )

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_user_token(uid, creds)
            print(f"[INFO] Drive token refreshed for user {uid}")
        except Exception as e:
            print(f"[WARN] Failed to refresh Drive token for user {uid}: {e}")

    return creds


def is_connected(user_id: str | None = None) -> bool:
    uid = str(user_id or "").strip()
    if not uid:
        return False
    from src.database import Database
    db = Database.get_instance()
    raw = db.get_user_drive_token(uid)
    return bool(raw and raw.get("token"))


def disconnect(user_id: str | None = None) -> bool:
    uid = str(user_id or "").strip()
    if not uid:
        return False
    from src.database import Database
    db = Database.get_instance()
    ok = db.delete_user_drive_token(uid)
    print(f"[INFO] Drive token deleted from Supabase for user {uid}: {ok}")
    return ok


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
