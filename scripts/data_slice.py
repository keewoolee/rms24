from __future__ import annotations

import hashlib
import json
from pathlib import Path

ACCOUNT_RECORD_SIZE = 24
STORAGE_RECORD_SIZE = 56


def filter_account_mapping_bytes(data: bytes, max_index: int, out_path: Path) -> None:
    if len(data) % ACCOUNT_RECORD_SIZE != 0:
        raise ValueError("account mapping data length is not aligned to record size")
    with out_path.open("wb") as out:
        for i in range(0, len(data), ACCOUNT_RECORD_SIZE):
            record = data[i : i + ACCOUNT_RECORD_SIZE]
            idx = int.from_bytes(record[20:24], "little")
            if idx < max_index:
                out.write(record)


def filter_storage_mapping_bytes(data: bytes, max_index: int, out_path: Path) -> None:
    if len(data) % STORAGE_RECORD_SIZE != 0:
        raise ValueError("storage mapping data length is not aligned to record size")
    with out_path.open("wb") as out:
        for i in range(0, len(data), STORAGE_RECORD_SIZE):
            record = data[i : i + STORAGE_RECORD_SIZE]
            idx = int.from_bytes(record[52:56], "little")
            if idx < max_index:
                out.write(record)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(meta_path: Path, entries: int, entry_size: int, files: dict[str, Path], source_tag: str) -> None:
    meta = {
        "entries": entries,
        "entry_size": entry_size,
        "source": source_tag,
        "files": {
            name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
