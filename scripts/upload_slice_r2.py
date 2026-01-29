from __future__ import annotations

import argparse
from pathlib import Path


def build_object_keys(prefix: str, files: list[str]) -> dict[str, str]:
    return {name: f"{prefix}/{name}" for name in files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--prefix", default="mainnet-v3-slice-1m")
    parser.add_argument("--bucket", default="pir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = ["database.bin", "account-mapping.bin", "storage-mapping.bin", "metadata.json"]
    keys = build_object_keys(args.prefix, files)

    if args.dry_run:
        for name, key in keys.items():
            print(f"DRY RUN: {name} -> s3://{args.bucket}/{key}")
        return

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url="https://pir.53627.org",
    )
    for name, key in keys.items():
        s3.upload_file(str(Path(args.dir) / name), args.bucket, key)


if __name__ == "__main__":
    main()
