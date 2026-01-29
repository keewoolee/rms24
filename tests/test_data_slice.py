import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import data_slice
from scripts.make_mainnet_slice import ENTRY_SIZE, copy_db_slice


def test_filter_account_mapping_by_index(tmp_path: Path):
    # 3 records: idx 5, idx 10, and idx 12
    record = lambda addr, idx: addr + idx.to_bytes(4, "little")
    data = record(b"a" * 20, 5) + record(b"b" * 20, 10) + record(b"c" * 20, 12)
    out = tmp_path / "account.bin"

    data_slice.filter_account_mapping_bytes(data, max_index=10, out_path=out)

    assert out.read_bytes() == record(b"a" * 20, 5)


def test_filter_account_mapping_excludes_max_index(tmp_path: Path):
    record = lambda addr, idx: addr + idx.to_bytes(4, "little")
    data = record(b"a" * 20, 10)
    out = tmp_path / "account.bin"

    data_slice.filter_account_mapping_bytes(data, max_index=10, out_path=out)

    assert out.read_bytes() == b""


def test_filter_storage_mapping_by_index(tmp_path: Path):
    record = lambda addr, slot, idx: addr + slot + idx.to_bytes(4, "little")
    data = record(b"a" * 20, b"s" * 32, 3) + record(b"b" * 20, b"t" * 32, 10) + record(b"c" * 20, b"u" * 32, 99)
    out = tmp_path / "storage.bin"

    data_slice.filter_storage_mapping_bytes(data, max_index=10, out_path=out)

    assert out.read_bytes() == record(b"a" * 20, b"s" * 32, 3)


def test_filter_account_mapping_invalid_length_raises(tmp_path: Path):
    data = b"a" * 23
    out = tmp_path / "account.bin"

    with pytest.raises(ValueError):
        data_slice.filter_account_mapping_bytes(data, max_index=10, out_path=out)


def test_filter_storage_mapping_invalid_length_raises(tmp_path: Path):
    data = b"a" * 55
    out = tmp_path / "storage.bin"

    with pytest.raises(ValueError):
        data_slice.filter_storage_mapping_bytes(data, max_index=10, out_path=out)


def test_filter_account_mapping_file(tmp_path: Path):
    record = lambda addr, idx: addr + idx.to_bytes(4, "little")
    data = record(b"a" * 20, 3) + record(b"b" * 20, 10) + record(b"c" * 20, 11)
    src = tmp_path / "account.bin"
    out = tmp_path / "account_out.bin"
    src.write_bytes(data)

    data_slice.filter_account_mapping_file(src, max_index=10, out_path=out, chunk_records=2)

    assert out.read_bytes() == record(b"a" * 20, 3)


def test_filter_storage_mapping_file(tmp_path: Path):
    record = lambda addr, slot, idx: addr + slot + idx.to_bytes(4, "little")
    data = record(b"a" * 20, b"s" * 32, 3) + record(b"b" * 20, b"t" * 32, 10)
    data += record(b"c" * 20, b"u" * 32, 11)
    src = tmp_path / "storage.bin"
    out = tmp_path / "storage_out.bin"
    src.write_bytes(data)

    data_slice.filter_storage_mapping_file(src, max_index=10, out_path=out, chunk_records=2)

    assert out.read_bytes() == record(b"a" * 20, b"s" * 32, 3)


def test_filter_account_mapping_file_invalid_length_raises(tmp_path: Path):
    src = tmp_path / "account.bin"
    out = tmp_path / "account_out.bin"
    src.write_bytes(b"a" * 23)

    with pytest.raises(ValueError):
        data_slice.filter_account_mapping_file(src, max_index=10, out_path=out)


def test_filter_storage_mapping_file_invalid_length_raises(tmp_path: Path):
    src = tmp_path / "storage.bin"
    out = tmp_path / "storage_out.bin"
    src.write_bytes(b"a" * 55)

    with pytest.raises(ValueError):
        data_slice.filter_storage_mapping_file(src, max_index=10, out_path=out)


def test_filter_account_mapping_file_invalid_chunk_records_raises(tmp_path: Path):
    record = lambda addr, idx: addr + idx.to_bytes(4, "little")
    data = record(b"a" * 20, 3)
    src = tmp_path / "account.bin"
    out = tmp_path / "account_out.bin"
    src.write_bytes(data)

    with pytest.raises(ValueError):
        data_slice.filter_account_mapping_file(src, max_index=10, out_path=out, chunk_records=0)


def test_filter_storage_mapping_file_invalid_chunk_records_raises(tmp_path: Path):
    record = lambda addr, slot, idx: addr + slot + idx.to_bytes(4, "little")
    data = record(b"a" * 20, b"s" * 32, 3)
    src = tmp_path / "storage.bin"
    out = tmp_path / "storage_out.bin"
    src.write_bytes(data)

    with pytest.raises(ValueError):
        data_slice.filter_storage_mapping_file(src, max_index=10, out_path=out, chunk_records=0)


def test_write_metadata(tmp_path: Path):
    db = tmp_path / "database.bin"
    payload_bytes = b"x" * 40
    db.write_bytes(payload_bytes)
    meta = tmp_path / "metadata.json"

    data_slice.write_metadata(
        meta_path=meta,
        entries=1,
        entry_size=40,
        files={"database.bin": db},
        source_tag="mainnet-v3",
    )

    payload = json.loads(meta.read_text())
    assert payload["entries"] == 1
    assert payload["entry_size"] == 40
    assert payload["source"] == "mainnet-v3"
    assert "database.bin" in payload["files"]
    assert payload["files"]["database.bin"]["bytes"] == 40
    assert payload["files"]["database.bin"]["sha256"] == hashlib.sha256(payload_bytes).hexdigest()


def test_copy_db_slice_requires_full_bytes(tmp_path: Path):
    source_db = tmp_path / "database.bin"
    source_db.write_bytes(b"a" * ENTRY_SIZE)
    out_db = tmp_path / "out.bin"

    with pytest.raises(ValueError):
        copy_db_slice(source_db, out_db, entries=2)

    assert not out_db.exists()


def test_make_slice_cli(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Create tiny mock DB (3 entries)
    (source_dir / "database.bin").write_bytes(b"a" * 40 + b"b" * 40 + b"c" * 40)
    # Account mapping: two records with indices 0 and 2
    (source_dir / "account-mapping.bin").write_bytes(
        b"x" * 20 + (0).to_bytes(4, "little") + b"y" * 20 + (2).to_bytes(4, "little")
    )
    # Storage mapping: one record with index 1
    (source_dir / "storage-mapping.bin").write_bytes(
        b"z" * 20 + b"s" * 32 + (1).to_bytes(4, "little")
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "2",
    ]
    subprocess.check_call(cmd)

    assert (out_dir / "database.bin").stat().st_size == 2 * 40
    assert (out_dir / "account-mapping.bin").stat().st_size == 24  # idx 0 only
    assert (out_dir / "storage-mapping.bin").stat().st_size == 56  # idx 1 only
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["entries"] == 2
    assert meta["entry_size"] == 40
    assert meta["files"]["database.bin"]["bytes"] == 2 * 40


def test_make_slice_cli_rejects_non_positive_entries(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "entries must be positive" in proc.stderr


def test_make_slice_cli_rejects_too_small_database(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "database.bin size" in proc.stderr


def test_make_slice_cli_rejects_non_integer_entries(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "abc",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "entries must be an integer" in proc.stderr


def test_make_slice_cli_rejects_missing_mapping_file(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "storage-mapping.bin" in proc.stderr


def test_make_slice_cli_rejects_missing_database(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 20 + (0).to_bytes(4, "little"))
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 20 + b"s" * 32 + (0).to_bytes(4, "little"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "database.bin" in proc.stderr


def test_make_slice_cli_rejects_unaligned_account_mapping(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 23)
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 56)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "account-mapping.bin size" in proc.stderr


def test_make_slice_cli_rejects_unaligned_storage_mapping(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 24)
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 55)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "storage-mapping.bin size" in proc.stderr


def test_make_slice_cli_cleans_outputs_on_failure(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 23)
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 56)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert not (out_dir / "database.bin").exists()
    assert not (out_dir / "account-mapping.bin").exists()
    assert not (out_dir / "storage-mapping.bin").exists()
    assert not (out_dir / "metadata.json").exists()


def test_make_slice_cli_rejects_out_inside_source(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_mainnet_slice.py"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "database.bin").write_bytes(b"a" * 40)
    (source_dir / "account-mapping.bin").write_bytes(b"x" * 24)
    (source_dir / "storage-mapping.bin").write_bytes(b"z" * 56)
    out_dir = source_dir / "out"

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--entries",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "output directory must be outside" in proc.stderr


def test_download_helper_validates_checksum(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    data = b"abc"
    file_path = out_dir / "database.bin"
    file_path.write_bytes(data)

    meta = {
        "entries": 1,
        "entry_size": 40,
        "files": {
            "database.bin": {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": 3,
            }
        },
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta))

    from scripts import download_slice

    download_slice.verify_files(out_dir, meta)


def test_build_r2_keys():
    from scripts import upload_slice_r2

    files = ["database.bin", "metadata.json"]
    keys = upload_slice_r2.build_object_keys(prefix="mainnet-v3-slice-1m", files=files)

    assert keys["database.bin"].endswith("mainnet-v3-slice-1m/database.bin")
