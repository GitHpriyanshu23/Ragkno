"""Compare local/server AgentRouter access without printing credentials or documents."""

import argparse
import hashlib
import os
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file, override=True)
    key = os.getenv("AGENTROUTER_API_KEY") or os.getenv("AGENT_ROUTER_API_KEY") or ""
    base = os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1").rstrip("/")
    model = os.getenv("AGENTROUTER_MODEL", "deepseek-v4-flash")
    if not key:
        raise SystemExit("AgentRouter key is missing in this environment.")
    print("Endpoint:", base)
    print("Model:", model)
    print("Key fingerprint (compare only):", hashlib.sha256(key.encode()).hexdigest()[:12])
    try:
        with requests.post(
            base + "/chat/completions",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "Accept": "text/event-stream", "User-Agent": "codex_cli_rs/0.1.0", "x-app": "cli"},
            json={"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "stream": True},
            stream=True, timeout=(10, 45),
        ) as response:
            print("HTTP status:", response.status_code)
            print("Final host:", urlsplit(response.url).hostname)
            content_type = response.headers.get("Content-Type", "missing")
            print("Content-Type:", content_type)
            # Only inspect a bounded prefix; never print the response body.
            prefix = next(response.iter_content(chunk_size=8192), b"").decode("utf-8", errors="replace").lower()
            challenge = any(marker in prefix for marker in (
                "cf_app_waf", "captcha", "challenge-platform", "just a moment",
                "verify you are human", "aliyun_waf", "ac_opt",
            ))
            print("Security challenge detected:", challenge)
            if "text/html" in content_type.lower():
                print("FAIL: Received an HTML page instead of a model API response.")
            elif response.status_code in (401, 403):
                print("FAIL: Provider rejected access; verify the matching endpoint, key and permissions.")
            elif not response.ok:
                print("FAIL: Provider returned an HTTP error.")
            else:
                print("API format received. This is a connectivity check, not full answer verification.")
    except requests.RequestException as error:
        # Avoid exception strings that could include response bodies or secrets.
        print("FAIL: Network request raised", type(error).__name__)


if __name__ == "__main__":
    main()
