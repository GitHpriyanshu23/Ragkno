"""
backend/main.py

FastAPI backend exposing RAG pipeline and Google Drive OAuth endpoints.
"""

import os
import uuid
import json
import pickle
import base64
import hashlib
import hmac
import time
import math
import secrets
import threading
import re
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel, Field
import bcrypt

# Always load env relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Allow OAuth over HTTP on localhost during development
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

# Patch sys.path so we can import from `src/`
import sys

class _SafeStream:
    def __init__(self, target):
        self._target = target

    def write(self, data):
        try:
            if self._target:
                return self._target.write(data)
        except (OSError, IOError):
            return len(data) if data else 0

    def flush(self):
        try:
            if self._target:
                self._target.flush()
        except (OSError, IOError):
            pass

    def isatty(self):
        try:
            return self._target.isatty() if self._target else False
        except Exception:
            return False

    def fileno(self):
        try:
            return self._target.fileno()
        except Exception:
            raise OSError(5, "Input/output error")

    def __getattr__(self, name):
        return getattr(self._target, name)

sys.stdout = _SafeStream(sys.stdout)
sys.stderr = _SafeStream(sys.stderr)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drive_loader import (
    disconnect,
    exchange_code,
    get_auth_url,
    is_connected,
    list_drive_files,
    load_credentials,
    load_documents_from_drive,
    pop_oauth_verifier,
    save_oauth_verifier,
)
from src.chroma_store import ChromaVectorStore
from src.database import Database
from src.chat_memory import ChatMemoryStore
from src.ingest import MAX_UPLOAD_BYTES, load_uploaded_file, load_url_content

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(os.getenv("RAGKNO_DATA_DIR", str(_PROJECT_ROOT))).expanduser()
STORE_DIR = str(
    Path(
        os.getenv(
            "CHROMA_PERSIST_DIR",
            str(_DATA_ROOT / "chroma_store"),
        )
    ).expanduser()
)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")
SESSION_COOKIE_NAME = "ragkno_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
APP_LOGIN_REDIRECT_URI = os.getenv("GOOGLE_APP_REDIRECT_URI", "http://localhost:8000/login/google/callback")
SESSION_SECRET = os.getenv("RAGKNO_SESSION_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET") or "ragkno-dev-session-secret"

if os.getenv("ENV", "").lower() in {"production", "prod"}:
    if SESSION_SECRET == "ragkno-dev-session-secret" or len(SESSION_SECRET) < 32:
        raise RuntimeError("RAGKNO_SESSION_SECRET must be set to at least 32 characters in production")
    if not FRONTEND_URL.startswith("https://"):
        raise RuntimeError("FRONTEND_URL must use HTTPS in production")
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL must be configured in production")
    if not (
        os.getenv("AGENTROUTER_API_KEY")
        or os.getenv("AGENT_ROUTER_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ):
        raise RuntimeError("Configure AGENTROUTER_API_KEY or GOOGLE_API_KEY in production")

app = FastAPI(title="RAG API", version="1.0.0")

_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]
if FRONTEND_URL and FRONTEND_URL not in _allowed_origins:
    _allowed_origins.append(FRONTEND_URL)

_extra_origins = os.getenv("CORS_ORIGINS", "")
if _extra_origins:
    for _o in _extra_origins.split(","):
        _o = _o.strip()
        if _o and _o not in _allowed_origins:
            _allowed_origins.append(_o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_db = Database.get_instance()
_store_instance = None
_rag_search_instance = None
_runtime_lock = threading.RLock()
_login_attempts: dict[str, deque[float]] = defaultdict(deque)
_login_attempts_lock = threading.Lock()
_LOGIN_WINDOW_SECONDS = 15 * 60
_LOGIN_ATTEMPT_LIMIT = 8

def _get_store() -> ChromaVectorStore:
    global _store_instance
    with _runtime_lock:
        if _store_instance is None:
            _store_instance = ChromaVectorStore(persist_dir=STORE_DIR)
    return _store_instance


def _get_rag():
    global _rag_search_instance
    with _runtime_lock:
        if _rag_search_instance is None:
            from src.search import RAGSearch
            _rag_search_instance = RAGSearch(persist_dir=STORE_DIR)
        return _rag_search_instance


def _invalidate_retrieval_cache() -> None:
    global _rag_search_instance
    with _runtime_lock:
        if _rag_search_instance is not None:
            _rag_search_instance.invalidate_caches()



def _build_login_flow() -> Flow:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth credentials are not configured.")

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": [APP_LOGIN_REDIRECT_URI],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        ],
        redirect_uri=APP_LOGIN_REDIRECT_URI,
    )


def _sign_payload(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    signature = hmac.new(SESSION_SECRET.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _verify_session_token(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    expected = hmac.new(SESSION_SECRET.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        padded = encoded + ("=" * (-len(encoded) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))
    except Exception:
        return None
    if int(payload.get("exp", 0) or 0) < int(time.time()):
        return None
    return payload


def _session_user_from_request(request: Request) -> dict | None:
    payload = _verify_session_token(request.cookies.get(SESSION_COOKIE_NAME))
    if not payload:
        return None
    return {
        "id": payload.get("sub"),
        "email": payload.get("email"),
        "name": payload.get("name") or payload.get("email") or "RagKno user",
        "picture": payload.get("picture"),
        "provider": payload.get("provider"),
        "csrf": payload.get("csrf"),
    }


def _is_prod_cookie() -> bool:
    return FRONTEND_URL.startswith("https://") or os.getenv("ENV", "").lower() in {"production", "prod"}


def _set_session_cookie(response: Response, user_info: dict) -> None:
    now = int(time.time())
    token = _sign_payload({
        "sub": user_info.get("sub") or user_info.get("id"),
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
        "provider": user_info.get("provider") or user_info.get("auth_provider") or "google",
        "iat": now,
        "exp": now + SESSION_TTL_SECONDS,
        "csrf": secrets.token_urlsafe(32),
    })
    is_prod = _is_prod_cookie()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="none" if is_prod else "lax",
        secure=is_prod,
        path="/",
    )


def _require_user(request: Request, mutation: bool = False) -> dict:
    user = _session_user_from_request(request)
    if not user or not user.get("id"):
        raise HTTPException(status_code=401, detail="Authentication required.")
    if mutation:
        origin = (request.headers.get("origin") or "").rstrip("/")
        if not origin or origin not in _allowed_origins:
            raise HTTPException(status_code=403, detail="Origin is not allowed.")
        supplied = request.headers.get("x-csrf-token") or ""
        expected = str(user.get("csrf") or "")
        if not expected or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=403, detail="Invalid CSRF token.")
    return user


def _require_public_auth_origin(request: Request) -> None:
    """Password endpoints are cookie-adjacent, so only accept our browser origins."""
    origin = (request.headers.get("origin") or "").rstrip("/")
    if not origin or origin not in _allowed_origins:
        raise HTTPException(status_code=403, detail="Origin is not allowed.")


def _normalized_email(value: str) -> str:
    email = value.strip().casefold()
    if len(email) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return email


def _validate_password(password: str) -> None:
    if len(password) < 12 or len(password.encode("utf-8")) > 72:
        raise HTTPException(status_code=422, detail="Password must be 12–72 bytes long.")
    if not all((re.search(pattern, password) for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^\w\s]"))):
        raise HTTPException(status_code=422, detail="Password must include upper and lower case letters, a number, and a symbol.")


def _login_attempt_key(request: Request, email: str) -> str:
    client = request.client.host if request.client else "unknown"
    return f"{client}:{email}"


def _allow_login_attempt(key: str) -> bool:
    now = time.time()
    with _login_attempts_lock:
        attempts = _login_attempts[key]
        while attempts and now - attempts[0] > _LOGIN_WINDOW_SECONDS:
            attempts.popleft()
        return len(attempts) < _LOGIN_ATTEMPT_LIMIT


def _record_failed_login(key: str) -> None:
    with _login_attempts_lock:
        _login_attempts[key].append(time.time())


def _clear_login_attempts(key: str) -> None:
    with _login_attempts_lock:
        _login_attempts.pop(key, None)


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------


@app.get("/app-auth/google/url", summary="Get Google app-login consent URL")
@app.get("/login/google/url", summary="Get Google app-login consent URL")
def login_google_url():
    flow = _build_login_flow()
    auth_url, state = flow.authorization_url(
        access_type="online",
        include_granted_scopes="true",
        prompt="select_account",
    )
    if not flow.code_verifier:
        raise HTTPException(status_code=500, detail="Failed to initialize login verifier.")
    save_oauth_verifier(state, flow.code_verifier)
    return {"url": auth_url}


@app.get("/login/google/callback", summary="Google app-login callback")
def login_google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    if error:
        raise HTTPException(
            status_code=400,
            detail=f"Google login error: {error}. {error_description or ''}".strip(),
        )
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing authorization code in login callback.")

    flow = _build_login_flow()
    code_verifier = pop_oauth_verifier(state)
    if not code_verifier:
        raise HTTPException(status_code=400, detail="OAuth state is invalid or expired.")
    flow.code_verifier = code_verifier

    try:
        flow.fetch_token(code=code)
        token = flow.credentials.id_token
        if not token:
            raise RuntimeError("Google did not return an identity token.")
        user_info = id_token.verify_oauth2_token(
            token,
            GoogleAuthRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
        )
        stored_user = _db.upsert_user(user_info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google login failed: {e}") from e

    response = RedirectResponse(url=f"{FRONTEND_URL}/chat")
    _set_session_cookie(response, stored_user)
    return response


@app.get("/auth/me", summary="Current logged-in app user")
def auth_me(request: Request, response: Response):
    response.headers["Cache-Control"] = "private, no-store"
    user = _session_user_from_request(request)
    if not user:
        return {"authenticated": False, "user": None}
    user_dict = dict(user)
    csrf_token = user_dict.pop("csrf", None)
    if user.get("picture"):
        user_dict["avatar_url"] = f"/auth/avatar?url={quote(str(user['picture']), safe='')}"
    return {"authenticated": True, "user": user_dict, "csrf_token": csrf_token}


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=256)
    terms_accepted: bool


class PasswordLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


def _auth_response(user: dict) -> dict:
    return {
        "authenticated": True,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("name") or user["email"],
            "picture": user.get("picture") or None,
            "provider": user.get("provider") or user.get("auth_provider") or "password",
        },
    }


@app.post("/auth/register", summary="Create a password account and signed session")
def auth_register(payload: RegisterRequest, request: Request, response: Response):
    _require_public_auth_origin(request)
    if not payload.terms_accepted:
        raise HTTPException(status_code=422, detail="You must accept the Terms of Service.")
    email = _normalized_email(payload.email)
    _validate_password(payload.password)
    name = " ".join(payload.name.split())
    if len(name) < 2:
        raise HTTPException(status_code=422, detail="Enter your name.")
    attempt_key = _login_attempt_key(request, email)
    if not _allow_login_attempt(attempt_key):
        raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")
    try:
        rounds = max(10, min(int(os.getenv("BCRYPT_ROUNDS", "12")), 14))
        password_hash = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt(rounds=rounds)).decode("utf-8")
        user = _db.create_password_user(name, email, password_hash)
    except ValueError:
        _record_failed_login(attempt_key)
        raise HTTPException(status_code=409, detail="An account already exists for this email. Sign in instead.")
    _clear_login_attempts(attempt_key)
    _set_session_cookie(response, user)
    return _auth_response(user)


@app.post("/auth/login", summary="Sign in with email and password")
def auth_password_login(payload: PasswordLoginRequest, request: Request, response: Response):
    _require_public_auth_origin(request)
    email = _normalized_email(payload.email)
    attempt_key = _login_attempt_key(request, email)
    if not _allow_login_attempt(attempt_key):
        raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")
    user = _db.get_password_user_by_email(email)
    password_hash = str(user.get("password_hash") or "") if user else ""
    valid = False
    try:
        valid = bool(password_hash) and bcrypt.checkpw(payload.password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        _record_failed_login(attempt_key)
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    _clear_login_attempts(attempt_key)
    _db.record_password_login(user["id"])
    _set_session_cookie(response, user)
    return _auth_response(user)


@app.get("/auth/avatar", summary="Proxy active user Google avatar image")
async def auth_avatar(request: Request, url: str | None = None):
    target_url = None
    if url and url.startswith("https://lh3.googleusercontent.com/"):
        target_url = url
    else:
        user = _session_user_from_request(request)
        if user and user.get("picture"):
            target_url = user["picture"]

    if not target_url:
        raise HTTPException(status_code=404, detail="No avatar found.")

    import urllib.request
    try:
        req = urllib.request.Request(
            target_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read()
            media_type = resp.headers.get("content-type", "image/jpeg")
            return Response(
                content=content,
                media_type=media_type,
                headers={
                    "Cache-Control": "private, no-store",
                    "Access-Control-Allow-Origin": "*",
                },
            )
    except Exception as e:
        return RedirectResponse(url=target_url, headers={"Cache-Control": "private, no-store"})



@app.post("/auth/logout", summary="Log out current app user")
def auth_logout(request: Request, response: Response):
    _require_user(request, mutation=True)
    is_prod = _is_prod_cookie()
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        samesite="none" if is_prod else "lax",
        secure=is_prod,
    )
    return {"ok": True, "message": "Logged out."}



# ---------------------------------------------------------------------------
# Thread & Message routes (per-user ChatGPT-style chat management)
# ---------------------------------------------------------------------------


class CreateThreadRequest(BaseModel):
    title: str = "New Chat"
    id: str | None = None


class RenameThreadRequest(BaseModel):
    title: str


@app.get("/threads", summary="Get all chat threads for the authenticated user")
def get_threads(request: Request):
    user = _require_user(request)
    threads = _db.get_user_threads(user["id"])
    return {"threads": threads}


@app.post("/threads", summary="Create a new chat thread")
def create_thread(req: CreateThreadRequest, request: Request):
    user = _require_user(request, mutation=True)
    thread = _db.create_thread(user_id=user["id"], title=req.title, thread_id=req.id)
    return {"thread": thread}


@app.get("/threads/{thread_id}/messages", summary="Get all messages for a thread")
def get_thread_messages(thread_id: str, request: Request):
    user = _require_user(request)
    if not _db.get_thread(thread_id, user["id"]):
        raise HTTPException(status_code=404, detail="Thread not found.")
    messages = _db.get_thread_messages(thread_id, user_id=user["id"])
    return {"messages": messages}


@app.patch("/threads/{thread_id}", summary="Rename a thread")
def rename_thread(thread_id: str, req: RenameThreadRequest, request: Request):
    user = _require_user(request, mutation=True)
    ok = _db.rename_thread(thread_id, user["id"], req.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return {"ok": True}


@app.delete("/threads/{thread_id}", summary="Delete a thread and its messages")
def delete_thread(thread_id: str, request: Request):
    user = _require_user(request, mutation=True)
    ok = _db.delete_thread(thread_id, user["id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return {"ok": True}


@app.get("/auth/url", summary="Get Google OAuth consent URL")
def auth_url(request: Request):
    user = _require_user(request)
    url = get_auth_url(user_id=user["id"])
    return {"url": url}


@app.get("/auth/callback", summary="OAuth callback — exchange code and redirect")
def auth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    # If Google sends an explicit OAuth error, redirect back to frontend with error param
    if error:
        return RedirectResponse(url=f"{FRONTEND_URL}/chat/data?error={error}")

    if not code:
        return RedirectResponse(url=f"{FRONTEND_URL}/chat/data?error=missing_code")

    user = _require_user(request)
    user_id = user["id"]

    try:
        if not state:
            raise ValueError("missing_state")
        exchange_code(code, state=state, user_id=user_id)
    except Exception as e:
        print(f"[ERROR] Token exchange failed: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}/chat/data?error=token_exchange_failed")
    # Redirect browser back to the React app with a success flag
    return RedirectResponse(url=f"{FRONTEND_URL}/chat/data?connected=1")


@app.get("/auth/status", summary="Check if Drive is connected for active user")
def auth_status(request: Request):
    user = _require_user(request)
    user_id = user["id"]
    return {"connected": is_connected(user_id)}


# ---------------------------------------------------------------------------
# Drive routes
# ---------------------------------------------------------------------------


@app.get("/drive/files", summary="List PDF/TXT files in Google Drive for active user")
def drive_files(request: Request):
    user = _require_user(request)
    user_id = user["id"]
    creds = load_credentials(user_id)
    if not creds:
        raise HTTPException(status_code=401, detail="Not connected to Google Drive.")
    try:
        files = list_drive_files(creds)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DriveSyncRequest(BaseModel):
    file_ids: list[str] | None = None


class UrlIngestRequest(BaseModel):
    url: str


class UnindexRequest(BaseModel):
    source_id: str | None = None
    key: str | None = None


def _source_result(item: dict, status: str = "indexed", error: str | None = None) -> dict:
    return {
        "source_id": item.get("source_id"),
        "display_name": item.get("source_name") or item.get("source") or item.get("source_key"),
        "status": status,
        "page_count": int(item.get("page_count", 0) or 0),
        "chunk_count": int(item.get("chunk_count", 0) or 0),
        "error": error,
    }


def _load_index_metadata() -> list[dict]:
    try:
        return _get_store().metadata
    except Exception:
        return []


def _source_type(source: str) -> str:
    lower = source.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        return "url"
    if lower.startswith("drive://") or "drive.google.com" in lower:
        return "drive"
    return "upload"


@app.get("/ingest/sources", summary="List unique indexed sources for active user")
def ingest_sources(request: Request):
    user = _require_user(request)
    user_id = user["id"]

    db_sources = _db.get_user_sources(user_id)
    chroma_sources = _get_store().get_user_sources(user_id=user_id)
    # Database rows are newest first; vector-store iteration has no time order.
    source_order = {str(s.get("source_key") or ""): index for index, s in enumerate(db_sources)}
    display_order = {}

    seen = set()
    seen_display = set()
    sources = []

    for s in chroma_sources:
        source_id = str(s.get("source_id") or "")
        source_key = str(s.get("source_key") or s.get("source") or "")
        if source_id and source_key not in seen:
            display_order[source_id] = source_order.get(source_key, len(db_sources))
            seen.add(source_key)
            seen_display.add(str(s.get("title") or s.get("source") or source_key).casefold())
            sources.append({
                "key": source_id,
                "source_id": source_id,
                "source": s.get("title") or s.get("source") or source_key,
                "type": s.get("source_type") or _source_type(source_key),
                "chunk_count": s.get("chunk_count", 0),
            })

    for s in db_sources:
        source_key = str(s.get("source_key") or "")
        source_id = str(s.get("id") or "")
        display_name = str(s.get("source_name") or source_key)
        if source_id and source_key not in seen and display_name.casefold() not in seen_display:
            display_order[source_id] = source_order.get(source_key, len(db_sources))
            seen.add(source_key)
            seen_display.add(display_name.casefold())
            sources.append({
                "key": source_id,
                "source_id": source_id,
                "source": s.get("source_name") or source_key,
                "type": s.get("source_type") or _source_type(source_key),
                "chunk_count": s.get("chunk_count", 0),
            })

    sources.sort(key=lambda source: display_order[source["key"]])
    return {"count": len(sources), "sources": sources}


@app.post("/ingest/unindex", summary="Remove one indexed source from Chroma store")
def ingest_unindex(req: UnindexRequest, request: Request):
    user = _require_user(request, mutation=True)
    key = str(req.source_id or req.key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Source key is required.")

    user_id = user["id"]

    try:
        store = _get_store()
        matching = next((item for item in store.get_user_sources(user_id) if item.get("source_id") == key), None)
        removed_chunks = store.remove_source(key, user_id=user_id)
        _db.delete_user_source(user_id, key)
        if matching and matching.get("source_key"):
            _db.delete_user_source(user_id, str(matching["source_key"]))
        _invalidate_retrieval_cache()

        return {
            "ok": True,
            "removed": removed_chunks,
            "message": f"Unindexed source successfully ({removed_chunks} chunk(s) removed).",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drive/sync", summary="Ingest selected Drive files into Chroma vector store")
def drive_sync(request: Request, req: DriveSyncRequest | None = None):
    user = _require_user(request, mutation=True)
    return _index_drive(user, req)


def _index_drive(user, req, progress=None):
    user_id = user["id"]
    if progress:
        progress("Downloading selected Drive documents", 3)
    creds = load_credentials(user_id)
    if not creds:
        raise HTTPException(status_code=401, detail="Not connected to Google Drive.")
    try:
        selected_file_ids = req.file_ids if req else None
        docs = load_documents_from_drive(creds, file_ids=selected_file_ids)
        if not docs:
            if selected_file_ids:
                return {
                    "message": "No extractable content found in the selected files.",
                    "count": 0,
                }
            return {"message": "No supported documents found in Drive.", "count": 0}

        store = _get_store()
        indexed = store.add_documents(docs, user_id=user_id, progress=progress)
        for item in indexed:
            _db.record_user_source(user_id, source_key=item["source_key"], source_name=item["source_name"], source_type="drive", chunk_count=item["chunk_count"])
        _invalidate_retrieval_cache()

        if selected_file_ids is not None:
            return {
                "message": f"Synced {len(docs)} document pages from selected files into the vector store.",
                "count": sum(item["chunk_count"] for item in indexed),
                "selected_files": len(selected_file_ids),
                "sources": [_source_result(item) for item in indexed],
            }

        return {"message": f"Synced {len(indexed)} source(s) into the vector store.", "count": sum(item["chunk_count"] for item in indexed), "sources": [_source_result(item) for item in indexed]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/drive/disconnect", summary="Disconnect Google Drive for active user")
def drive_disconnect(request: Request):
    user = _require_user(request, mutation=True)
    user_id = user["id"]
    disconnect(user_id)
    return {"message": "Disconnected from Google Drive."}


@app.post("/ingest/files", summary="Upload local files and index into Chroma")
def ingest_files(request: Request, files: list[UploadFile] = File(...)):
    # Parsing and embedding are blocking work. A synchronous route runs in
    # FastAPI's worker pool so uploads cannot stall session and health requests.
    user = _require_user(request, mutation=True)
    return _index_uploaded_files(user, files)


def _index_uploaded_files(user, files, progress=None):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="A maximum of 20 files can be uploaded at once.")
    allowed_mimes = {
        ".pdf": {"application/pdf"},
        ".txt": {"text/plain"},
        ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    }
    documents = []
    preliminary = []
    seen_names = set()
    for file in files:
        name = (file.filename or "uploaded-file").strip()
        suffix = Path(name).suffix.lower()
        result = {"source_id": None, "display_name": name, "status": "failed", "page_count": 0, "chunk_count": 0, "error": None}
        if name in seen_names:
            result["error"] = "Duplicate filename in this upload"
            preliminary.append(result)
            continue
        seen_names.add(name)
        if suffix not in allowed_mimes:
            result["error"] = f"Unsupported file extension: {suffix or 'unknown'}"
            preliminary.append(result)
            continue
        content_type = (file.content_type or "").split(";", 1)[0].lower()
        if content_type not in {"", "application/octet-stream", *allowed_mimes[suffix]}:
            result["error"] = f"Content type '{content_type}' does not match {suffix}"
            preliminary.append(result)
            continue
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            result["error"] = "File exceeds the 50MB limit"
            preliminary.append(result)
            continue
        try:
            file_docs = load_uploaded_file(name, content)
            if not file_docs:
                raise ValueError("File is empty or contains no extractable text")
            documents.extend(file_docs)
            result["status"] = "ready"
            result["page_count"] = len(file_docs)
        except Exception as error:
            result["error"] = str(error)
        preliminary.append(result)

    try:
        if not documents:
            return {"ok": False, "message": "No files contained indexable text.", "indexed_chunks": 0, "source_count": 0, "sources": preliminary}

        user_id = user["id"]

        if progress:
            progress("Loading document index", 8)
        store = _get_store()
        indexed = store.add_documents(documents, user_id=user_id, progress=progress)
        for item in indexed:
            _db.record_user_source(user_id, source_key=item["source_key"], source_name=item["source_name"], source_type="upload", chunk_count=item["chunk_count"])
        _invalidate_retrieval_cache()

        indexed_by_name = {str(item.get("source_name")): item for item in indexed}
        results = []
        for result in preliminary:
            item = indexed_by_name.get(result["display_name"])
            if item and result["status"] == "ready":
                results.append(_source_result(item))
            else:
                if result["status"] == "ready":
                    result["status"] = "failed"
                    result["error"] = "No chunks were produced"
                results.append(result)

        return {
            "ok": True,
            "message": f"Indexed {len(indexed)} of {len(files)} uploaded file(s).",
            "indexed_chunks": sum(item["chunk_count"] for item in indexed),
            "source_count": len(indexed),
            "sources": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from src.ingestion_jobs import IngestionJobs


def _process_ingestion_job(job, directory, progress):
    from contextlib import ExitStack
    if job["kind"] == "drive":
        return _index_drive({"id": job["user_id"]}, DriveSyncRequest(**job["payload"]), progress=progress)
    from starlette.datastructures import Headers
    with ExitStack() as stack:
        files = [UploadFile(filename=item["name"], headers=Headers({"content-type": item["content_type"]}), file=stack.enter_context((directory / item["file"]).open("rb")))
                 for item in job["inputs"]]
        return _index_uploaded_files({"id": job["user_id"]}, files, progress=progress)


_ingestion_jobs = IngestionJobs(_DATA_ROOT / "ingestion_jobs", _process_ingestion_job)


@app.on_event("startup")
def resume_ingestion_jobs():
    _ingestion_jobs.start()


@app.on_event("shutdown")
def stop_ingestion_jobs():
    _ingestion_jobs.stop()


@app.post("/ingest/jobs/files", status_code=202, summary="Accept upload for background indexing")
def queue_uploaded_files(request: Request, files: list[UploadFile] = File(...)):
    user = _require_user(request, mutation=True)
    if not files or len(files) > 20:
        raise HTTPException(status_code=400, detail="Select between 1 and 20 files.")
    accepted = []
    total = 0
    for file in files:
        name = (file.filename or "uploaded-file").strip()
        if Path(name).suffix.lower() not in {".pdf", ".txt", ".docx"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {name}")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        total += len(content)
        if not content or len(content) > MAX_UPLOAD_BYTES or total > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Files must be non-empty, under 50MB each and 100MB total.")
        accepted.append((name, file.content_type or "", content))
    try:
        return _ingestion_jobs.submit(user["id"], accepted)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.post("/ingest/jobs/drive", status_code=202, summary="Index Drive files in background")
def queue_drive_sync(request: Request, req: DriveSyncRequest):
    user = _require_user(request, mutation=True)
    if not load_credentials(user["id"]):
        raise HTTPException(status_code=401, detail="Not connected to Google Drive.")
    try:
        return _ingestion_jobs.submit(user["id"], kind="drive", payload={"file_ids": req.file_ids})
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.get("/ingest/jobs", summary="Get active indexing jobs for the signed-in user")
def active_ingestion_jobs(request: Request, response: Response):
    response.headers["Cache-Control"] = "private, no-store"
    user = _require_user(request)
    return {"jobs": _ingestion_jobs.active(user["id"])}


@app.get("/ingest/jobs/{job_id}", summary="Get indexing progress")
def ingestion_job_status(job_id: str, request: Request, response: Response):
    user = _require_user(request)
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return _ingestion_jobs.get(job_id, user["id"])
    except KeyError:
        raise HTTPException(status_code=404, detail="Indexing job not found.")


@app.post("/ingest/url", summary="Fetch and index one website URL")
def ingest_url(req: UrlIngestRequest, request: Request):
    user = _require_user(request, mutation=True)
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    try:
        docs = load_url_content(req.url.strip())
        user_id = user["id"]

        store = _get_store()
        indexed = store.add_documents(docs, user_id=user_id)
        item = indexed[0]
        _db.record_user_source(user_id, source_key=item["source_key"], source_name=item["source_name"], source_type="url", chunk_count=item["chunk_count"])
        _invalidate_retrieval_cache()

        return {
            "ok": True,
            "message": "URL content indexed successfully.",
            "indexed_chunks": item["chunk_count"],
            "source_count": 1,
            "sources": [_source_result(item) for item in indexed],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Query route
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=20)
    thread_id: str = Field(min_length=1, max_length=160)
    request_id: str = Field(min_length=8, max_length=160)
    model: str | None = Field(default=None, min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:/-]+$")
    use_reranker: bool = True
    language: str | None = Field(default=None, max_length=32)
    source_ids: list[str] | None = Field(default=None, max_length=100)


_memory_store = ChatMemoryStore()


class SessionRequest(BaseModel):
    session_id: str


def _source_link(item: dict) -> str | None:
    source = str(item.get("source", "") or "")
    source_type = str(item.get("type", "") or "").lower()
    file_id = str(item.get("file_id", "") or "")

    if source_type == "url" and source:
        return source

    if source_type == "drive" and file_id:
        return f"https://drive.google.com/file/d/{file_id}/view"

    return None


def _prepare_sources_for_client(raw_sources: list[dict]) -> list[dict]:
    prepared = []
    for item in raw_sources or []:
        prepared.append(
            {
                "index": int(item.get("index", 0) or 0),
                "source": str(item.get("source", "") or ""),
                "type": str(item.get("type", "unknown") or "unknown"),
                "preview": str(item.get("preview", "") or ""),
                "text": str(item.get("text", "") or ""),
                "score": float(item.get("score", 0.0) or 0.0) if math.isfinite(float(item.get("score", 0.0) or 0.0)) else 0.0,
                "source_id": item.get("source_id"),
                "page": item.get("page"),
                "title": str(item.get("title", "") or ""),
                "file_id": item.get("file_id"),
                "link": _source_link(item),
            }
        )
    return prepared


def _maybe_refresh_summary(session_id: str, rag_search_instance, user_id: str = "default") -> None:
    turns_total = _memory_store.get_turn_count(session_id, user_id=user_id)
    if turns_total < 12 or turns_total % 4 != 0:
        return

    summary_record = _memory_store.get_summary_record(session_id, user_id=user_id)
    unsummarized = _memory_store.get_turns_since(session_id, summary_record["last_turn_id"], user_id=user_id)
    if len(unsummarized) <= 8:
        return

    summarize_now = unsummarized[:-6]
    if not summarize_now:
        return

    history_text = "\n".join(f"{item['role'].title()}: {item['text']}" for item in summarize_now)
    updated_summary = rag_search_instance.summarize_history(
        summary_record["summary_text"],
        history_text,
    )
    _memory_store.upsert_summary(session_id, updated_summary, summarize_now[-1]["id"], user_id=user_id)


def _refresh_summary_safely(session_id: str, rag_search_instance, user_id: str = "default") -> None:
    try:
        _maybe_refresh_summary(session_id, rag_search_instance, user_id=user_id)
    except Exception as error:
        print(f"[WARN] Conversation summary refresh failed: {error}")


def _sse_event(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n"


def _query_fingerprint(req: QueryRequest) -> str:
    payload = {
        "query": req.query.strip(),
        "thread_id": req.thread_id.strip(),
        "top_k": req.top_k,
        "model": req.model,
        "use_reranker": req.use_reranker,
        "language": req.language,
        "source_ids": sorted(set(req.source_ids or [])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_simple_greeting(value: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", " ", str(value or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in {
        "hi", "hello", "hey", "hiya", "howdy", "good morning",
        "good afternoon", "good evening", "hi there", "hello there", "hey there",
    }


def _local_greeting_answer(user: dict) -> str:
    display = str(user.get("name") or user.get("email") or "there").strip()
    first_name = display.split(" ")[0].split("@")[0] or "there"
    return (
        f"Hi {first_name}! Please provide a little more context about what you want to find, "
        "and I’ll search your connected documents for the most relevant information."
    )


@app.post("/memory/reset", summary="Reset chat memory for a session")
def reset_memory(req: SessionRequest, request: Request):
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    user = _require_user(request, mutation=True)
    user_id = user["id"]
    if not _db.get_thread(req.session_id.strip(), user_id):
        raise HTTPException(status_code=404, detail="Thread not found.")
    _memory_store.clear_session(req.session_id.strip(), user_id=user_id)
    return {"ok": True, "message": "Session memory cleared."}


@app.post("/query", summary="RAG query — retrieve and generate answer")
def handle_query(req: QueryRequest, request: Request):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    user = _require_user(request, mutation=True)
    user_id = user["id"]
    thread_id = req.thread_id.strip()
    if not _db.get_thread(thread_id, user_id):
        raise HTTPException(status_code=404, detail="Thread not found.")
    run = _db.begin_query_request(req.request_id, user_id, thread_id, _query_fingerprint(req))
    if run.get("status") == "complete":
        cached = _db.decode_query_request(run)
        return {"answer": cached.get("answer") or "", "query": req.query, "thread_id": thread_id, "request_id": req.request_id, "sources": cached.get("sources", []), "replayed": True, "local": _is_simple_greeting(req.query)}
    if run.get("status") != "created":
        raise HTTPException(status_code=409, detail="This request is already running.")
    try:
        if _is_simple_greeting(req.query):
            answer = _local_greeting_answer(user)
            _db.complete_query_exchange(req.request_id, thread_id, user_id, req.query.strip(), answer, [])
            return {
                "answer": answer,
                "query": req.query,
                "thread_id": thread_id,
                "request_id": req.request_id,
                "sources": [],
                "local": True,
            }

        rag = _get_rag()
        memory_context = _memory_store.build_memory_context(session_id=thread_id, recent_limit=8, user_id=user_id)
        response_payload = rag.answer_with_sources(
            query=req.query.strip(),
            top_k=req.top_k,
            memory_context=memory_context,
            user_id=user_id,
            model=req.model,
            use_reranker=req.use_reranker,
            source_ids=req.source_ids,
            language=req.language,
        )
        answer = response_payload.get("answer", "")
        sources = _prepare_sources_for_client(response_payload.get("sources", []))

        _db.complete_query_exchange(req.request_id, thread_id, user_id, req.query.strip(), answer, sources)
        try:
            _memory_store.append_turn(thread_id, "user", req.query.strip(), user_id=user_id)
            _memory_store.append_turn(thread_id, "assistant", answer, user_id=user_id)
            _maybe_refresh_summary(thread_id, rag, user_id=user_id)
        except Exception as memory_error:
            print(f"[WARN] Answer was saved but memory refresh failed: {memory_error}")

        return {
            "answer": answer,
            "query": req.query,
            "thread_id": thread_id,
            "request_id": req.request_id,
            "sources": sources,
        }
    except Exception as e:
        _db.complete_query_exchange(req.request_id, thread_id, user_id, req.query.strip(), "The response failed before it completed.", [], error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", summary="RAG query with server-sent events")
def handle_query_stream(req: QueryRequest, request: Request, background_tasks: BackgroundTasks):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    user = _require_user(request, mutation=True)
    user_id = user["id"]
    thread_id = req.thread_id.strip()
    if not _db.get_thread(thread_id, user_id):
        raise HTTPException(status_code=404, detail="Thread not found.")
    run = _db.begin_query_request(req.request_id, user_id, thread_id, _query_fingerprint(req))
    if run.get("status") not in {"created", "complete"}:
        raise HTTPException(status_code=409, detail="This request is already running.")

    def event_generator():
        answer_parts = []
        prepared_sources = []
        try:
            if run.get("status") == "complete":
                cached = _db.decode_query_request(run)
                answer = cached.get("answer") or ""
                sources = cached.get("sources") or []
                local = _is_simple_greeting(req.query)
                yield _sse_event("meta", {"query": req.query, "thread_id": thread_id, "request_id": req.request_id, "sources": sources, "replayed": True, "local": local})
                if answer:
                    if local:
                        time.sleep(0.18)
                        for chunk in re.findall(r"\S+\s*", answer):
                            yield _sse_event("token", {"request_id": req.request_id, "token": chunk})
                            time.sleep(0.025)
                    else:
                        yield _sse_event("token", {"request_id": req.request_id, "token": answer})
                yield _sse_event("done", {"query": req.query, "thread_id": thread_id, "request_id": req.request_id, "answer": answer, "sources": sources, "replayed": True, "local": local})
                return

            if _is_simple_greeting(req.query):
                answer = _local_greeting_answer(user)
                _db.complete_query_exchange(req.request_id, thread_id, user_id, req.query.strip(), answer, [])
                yield _sse_event("meta", {"query": req.query, "thread_id": thread_id, "request_id": req.request_id, "sources": [], "local": True})
                time.sleep(0.18)
                for chunk in re.findall(r"\S+\s*", answer):
                    yield _sse_event("token", {"request_id": req.request_id, "token": chunk})
                    time.sleep(0.025)
                yield _sse_event("done", {"query": req.query, "thread_id": thread_id, "request_id": req.request_id, "answer": answer, "sources": [], "local": True})
                return

            rag = _get_rag()
            memory_context = _memory_store.build_memory_context(session_id=thread_id, recent_limit=8, user_id=user_id)
            sources, token_iter = rag.stream_answer_with_sources(
                query=req.query.strip(),
                top_k=req.top_k,
                memory_context=memory_context,
                user_id=user_id,
                model=req.model,
                use_reranker=req.use_reranker,
                source_ids=req.source_ids,
                language=req.language,
            )
            prepared_sources = _prepare_sources_for_client(sources)
            yield _sse_event(
                "meta",
                {
                    "query": req.query,
                    "thread_id": thread_id,
                    "request_id": req.request_id,
                    "sources": prepared_sources,
                },
            )

            for token in token_iter:
                text = str(token or "")
                if not text:
                    continue
                answer_parts.append(text)
                yield _sse_event("token", {"request_id": req.request_id, "token": text})

            answer = "".join(answer_parts).strip() or "No relevant documents found."
            _db.complete_query_exchange(req.request_id, thread_id, user_id, req.query.strip(), answer, prepared_sources)
            try:
                _memory_store.append_turn(thread_id, "user", req.query.strip(), user_id=user_id)
                _memory_store.append_turn(thread_id, "assistant", answer, user_id=user_id)
                # Summarization may require another model call. Run it after
                # the response has completed so it never keeps the composer
                # in its stop/disabled state after the answer is visible.
                background_tasks.add_task(_refresh_summary_safely, thread_id, rag, user_id)
            except Exception as memory_error:
                print(f"[WARN] Answer was saved but memory refresh failed: {memory_error}")

            yield _sse_event(
                "done",
                {
                    "query": req.query,
                    "thread_id": thread_id,
                    "request_id": req.request_id,
                    "answer": answer,
                    "sources": prepared_sources,
                },
            )
        except Exception as e:
            _db.complete_query_exchange(
                req.request_id, thread_id, user_id, req.query.strip(),
                "".join(answer_parts).strip() or "The response failed before it completed.",
                prepared_sources, error=str(e),
            )
            yield _sse_event("error", {"request_id": req.request_id, "message": str(e), "interrupted": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ---------------------------------------------------------------------------
# Feedback routes
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    rating: str | None = "neutral"
    feedback: str


@app.post("/feedback", summary="Submit user feedback")
def submit_feedback(payload: FeedbackRequest, request: Request):
    user = _require_user(request, mutation=True)
    user_id = user["id"]
    feedback_text = (payload.feedback or "").strip()
    rating = (payload.rating or "neutral").strip() or "neutral"
    if not feedback_text:
        raise HTTPException(status_code=400, detail="Feedback cannot be empty.")
    saved = _db.save_feedback(user_id=user_id, rating=rating, feedback=feedback_text)
    return {"ok": True, "feedback": saved}


@app.get("/feedback", summary="List tracked feedback")
def list_feedbacks_route(request: Request, limit: int = 50):
    user = _require_user(request)
    feedbacks = _db.list_feedbacks(user_id=user["id"], limit=min(max(limit, 1), 100))
    return {"ok": True, "feedbacks": feedbacks}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    provider = (os.getenv("LLM_PROVIDER") or ("agentrouter" if (os.getenv("AGENTROUTER_API_KEY") or os.getenv("AGENT_ROUTER_API_KEY")) else "google")).strip().lower()
    return {
        "status": "ok",
        "llm": {
            "provider": provider,
            "model": os.getenv("AGENTROUTER_MODEL", "deepseek-v4-flash") if provider == "agentrouter" else os.getenv("GOOGLE_LLM_MODEL", "gemini-3.5-flash-lite"),
            "configured": bool(
                (os.getenv("AGENTROUTER_API_KEY") or os.getenv("AGENT_ROUTER_API_KEY"))
                if provider == "agentrouter"
                else os.getenv("GOOGLE_API_KEY")
            ),
        },
    }


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    checks = {"database": False, "vector_store": False, "llm_configured": False}
    try:
        with _db.engine.connect() as connection:
            from sqlalchemy import text
            connection.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        pass
    try:
        _get_store().collection.count()
        checks["vector_store"] = True
    except Exception:
        pass
    checks["llm_configured"] = health()["llm"]["configured"]
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}



if __name__ == "__main__":
    import uvicorn
    # This allows running via `python backend/main.py` and guarantees the local .venv is used
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
