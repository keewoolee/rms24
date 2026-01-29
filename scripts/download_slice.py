from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://pir.53627.org/mainnet-v3-slice-1m-mixed"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(out_dir: Path, meta: dict) -> None:
    for name, info in meta["files"].items():
        path = out_dir / name
        if path.stat().st_size != info["bytes"]:
            raise ValueError(f"Size mismatch for {name}")
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")


def download_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(f"{BASE_URL}/metadata.json", out_dir / "metadata.json")
    meta = json.loads((out_dir / "metadata.json").read_text())
    for name in meta["files"].keys():
        urlretrieve(f"{BASE_URL}/{name}", out_dir / name)
    verify_files(out_dir, meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    download_all(Path(args.out))


if __name__ == "__main__":
    main()
