"""
Tests for cuckoo hashing implementation.
"""

import pytest
import secrets
import sys

sys.path.insert(0, "..")

from rms24.cuckoo import CuckooParams, CuckooHash, CuckooTable


class TestCuckooParams:
    """Test parameter computation."""

    def test_basic_params(self):
        params = CuckooParams(
            key_size=16,
            value_size=32,
            num_hashes=2,
            num_buckets=3000,
        )
        assert params.key_size == 16
        assert params.value_size == 32
        assert params.num_hashes == 2
        assert params.num_buckets == 3000

    def test_entry_size(self):
        params = CuckooParams(
            key_size=8,
            value_size=24,
            num_hashes=2,
            num_buckets=120,
        )
        assert params.entry_size == 32  # 8 + 24

    def test_key_size_validation(self):
        with pytest.raises(ValueError, match="key_size must be at least 2"):
            CuckooParams(
                key_size=1,
                value_size=32,
                num_hashes=2,
                num_buckets=100,
            )

    def test_value_size_validation(self):
        with pytest.raises(ValueError, match="value_size must be at least 1"):
            CuckooParams(
                key_size=16,
                value_size=0,
                num_hashes=2,
                num_buckets=100,
            )

    def test_num_hashes_validation(self):
        with pytest.raises(ValueError, match="num_hashes must be at least 2"):
            CuckooParams(
                key_size=16,
                value_size=32,
                num_hashes=1,
                num_buckets=100,
            )

    def test_num_buckets_validation(self):
        with pytest.raises(ValueError, match="num_buckets must be at least 1"):
            CuckooParams(
                key_size=16,
                value_size=32,
                num_hashes=2,
                num_buckets=0,
            )



class TestCuckooHash:
    """Test hash functions."""

    def test_deterministic(self):
        hasher = CuckooHash(num_hashes=2, num_buckets=100, seed=b"0" * 16)
        key = b"test_key_1234567"
        h1 = hasher.hash(0, key)
        h2 = hasher.hash(0, key)
        assert h1 == h2

    def test_different_hash_indices(self):
        hasher = CuckooHash(num_hashes=2, num_buckets=1000, seed=b"0" * 16)
        key = b"test_key_1234567"
        positions = hasher.all_positions(key)
        # Different hash functions should (usually) give different positions
        # With 1000 buckets, collision is unlikely
        assert len(set(positions)) == 2

    def test_different_keys(self):
        hasher = CuckooHash(num_hashes=2, num_buckets=1000, seed=b"0" * 16)
        h1 = hasher.hash(0, b"key_aaaa_1234567")
        h2 = hasher.hash(0, b"key_bbbb_1234567")
        # Different keys should (usually) hash differently
        assert h1 != h2

    def test_range(self):
        hasher = CuckooHash(num_hashes=2, num_buckets=100, seed=b"0" * 16)
        for _ in range(100):
            key = secrets.token_bytes(16)
            for i in range(2):
                pos = hasher.hash(i, key)
                assert 0 <= pos < 100


class TestCuckooTable:
    """Test cuckoo hash table."""

    @pytest.fixture
    def small_params(self):
        return CuckooParams(
            key_size=16,
            value_size=32,
            num_hashes=2,
            num_buckets=150,
        )

    def test_insert_and_lookup(self, small_params):
        table = CuckooTable(small_params)
        key = b"k" * 16
        value = b"v" * 32
        table.insert(key, value)
        assert table.lookup(key) == value

    def test_lookup_missing(self, small_params):
        table = CuckooTable(small_params)
        key = b"k" * 16
        assert table.lookup(key) is None

    def test_multiple_inserts(self, small_params):
        table = CuckooTable(small_params)
        pairs = {}
        for i in range(30):
            key = f"key_{i:012d}".encode()
            value = f"value_{i:026d}".encode()
            pairs[key] = value
            table.insert(key, value)

        for key, value in pairs.items():
            assert table.lookup(key) == value

    def test_to_database(self, small_params):
        table = CuckooTable(small_params)
        key = b"k" * 16
        value = b"v" * 32
        table.insert(key, value)

        db = table.to_database()
        assert len(db) == small_params.num_buckets
        assert all(len(entry) == small_params.entry_size for entry in db)

        # One entry should be non-zero
        non_empty = [e for e in db if e != bytes(small_params.entry_size)]
        assert len(non_empty) == 1
        assert non_empty[0] == key + value

    def test_build(self, small_params):
        pairs = {
            f"key_{i:012d}".encode(): f"value_{i:026d}".encode()
            for i in range(30)
        }
        table = CuckooTable.build(pairs, small_params)

        for key, value in pairs.items():
            assert table.lookup(key) == value

    def test_stash_overflow(self):
        """Items that fail insertion go to stash."""
        params = CuckooParams(
            key_size=16,
            value_size=32,
            num_hashes=2,
            num_buckets=50,  # Way too small, will need stash
        )
        pairs = {
            f"key_{i:012d}".encode(): f"value_{i:026d}".encode()
            for i in range(100)
        }
        table = CuckooTable.build(pairs, params)

        # All items should be found (either in table or stash)
        for key, value in pairs.items():
            assert table.lookup(key) == value

        # Stash should be non-empty since table is overfull
        assert len(table.stash) > 0

    def test_wrong_key_size(self, small_params):
        table = CuckooTable(small_params)
        with pytest.raises(ValueError, match="Key size mismatch"):
            table.insert(b"short", b"v" * 32)

    def test_wrong_value_size(self, small_params):
        table = CuckooTable(small_params)
        with pytest.raises(ValueError, match="Value size mismatch"):
            table.insert(b"k" * 16, b"short")

    def test_seed_reproducibility(self, small_params):
        """Same seed should produce same hash positions."""
        seed = b"test_seed_123456"
        table1 = CuckooTable(small_params, seed=seed)
        table2 = CuckooTable(small_params, seed=seed)

        key = b"k" * 16
        pos1 = table1.hasher.all_positions(key)
        pos2 = table2.hasher.all_positions(key)
        assert pos1 == pos2

    def test_update(self, small_params):
        """Test updating an existing key in bucket."""
        table = CuckooTable(small_params)
        key = b"k" * 16
        value = b"v" * 32
        new_value = b"n" * 32
        table.insert(key, value)

        bucket_idx = table.update(key, new_value)
        assert bucket_idx is not None  # Key is in bucket, not stash
        assert table.lookup(key) == new_value

    def test_update_nonexistent_key(self, small_params):
        """Updating a nonexistent key raises KeyError."""
        table = CuckooTable(small_params)
        key = b"k" * 16
        value = b"v" * 32
        with pytest.raises(KeyError):
            table.update(key, value)

    def test_update_wrong_key_size(self, small_params):
        """Updating with wrong key size raises ValueError."""
        table = CuckooTable(small_params)
        with pytest.raises(ValueError, match="Key size mismatch"):
            table.update(b"short", b"v" * 32)

    def test_update_wrong_value_size(self, small_params):
        """Updating with wrong value size raises ValueError."""
        table = CuckooTable(small_params)
        key = b"k" * 16
        value = b"v" * 32
        table.insert(key, value)

        with pytest.raises(ValueError, match="Value size mismatch"):
            table.update(key, b"short")

    def test_update_stash_item(self):
        """Updating a stash item returns None."""
        # Use tiny table to force stash usage
        params = CuckooParams(
            key_size=16,
            value_size=32,
            num_hashes=2,
            num_buckets=10,
        )
        # Insert many items to force some into stash
        pairs = {
            f"key_{i:012d}".encode(): f"value_{i:026d}".encode()
            for i in range(50)
        }
        table = CuckooTable.build(pairs, params)
        assert len(table.stash) > 0, "Test requires items in stash"

        # Update a stash item
        stash_key, old_value = table.stash[0]
        new_value = b"updated_stash_value_________!!!!"  # 32 bytes
        bucket_idx = table.update(stash_key, new_value)

        assert bucket_idx is None
        assert table.lookup(stash_key) == new_value


class TestCuckooDefaultLoad:
    """Test cuckoo hashing with default parameters (κ=2, expansion=3)."""

    def test_default_load(self):
        """κ=2 with expansion_factor=3 should work reliably."""
        num_buckets = 1000 * 3

        params = CuckooParams(
            key_size=16,
            value_size=32,
            num_hashes=2,
            num_buckets=num_buckets,
        )
        pairs = {
            f"key_{i:012d}".encode(): f"value_{i:026d}".encode()
            for i in range(1000)
        }
        table = CuckooTable.build(pairs, params)

        # Verify all lookups
        for key, value in pairs.items():
            assert table.lookup(key) == value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
