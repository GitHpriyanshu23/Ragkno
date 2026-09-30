"""Back up Chroma and migrate legacy chunks to deterministic tenant-scoped IDs."""

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chroma_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default=str(ROOT / "chroma_store"))
    parser.add_argument("--backup-dir", default=str(ROOT / "backups"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    store_path = Path(args.store).resolve()
    if not args.apply:
        print("Dry run only. Pass --apply to create a backup and migrate legacy records.")
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Path(args.backup_dir).resolve() / f"chroma_store-{timestamp}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(store_path, backup)
    result = ChromaVectorStore(persist_dir=str(store_path), load_model=False).migrate_legacy_records()
    print({"backup": str(backup), **result})


if __name__ == "__main__":
    main()
