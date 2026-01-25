"""
Tests for Keyword PIR implementation.
"""

import pytest
import secrets
import sys

sys.path.insert(0, "..")

from pir.keyword_pir import KPIRParams, KPIRClient, KPIRServer
from pir.rms24 import Client as RMS24Client, Server as RMS24Server


def create_keyword_setup(
    num_items: int,
    key_size: int = 16,
    value_size: int = 32,
    security_param: int = 40,
    expansion_factor: int = 3,
):
    """Helper to create keyword PIR setup with RMS24 backend."""
    kw_params = KPIRParams(
        num_items_expected=num_items,
        key_size=key_size,
        value_size=value_size,
        expansion_factor=expansion_factor,
    )

    kv_pairs = {
        f"key_{i:012d}".encode(): f"value_{i:026d}".encode()
        for i in range(num_items)
    }

    server = KPIRServer.create(kv_pairs, kw_params, RMS24Server, security_param)
    client = KPIRClient.create(kw_params, RMS24Client, security_param)

    return kw_params, kv_pairs, server, client


class TestKPIRParams:
    """Test Keyword PIR parameter computation."""

    def test_basic_params(self):
        params = KPIRParams(
            num_items_expected=1000,
            key_size=16,
            value_size=32,
        )
        assert params.num_items_expected == 1000
        assert params.key_size == 16
        assert params.value_size == 32
        assert params.cuckoo_params.num_hashes == 2  # Default kappa
        assert params.cuckoo_params.num_buckets >= 1000

    def test_default_sizes(self):
        params = KPIRParams(num_items_expected=100)
        assert params.key_size == 32  # Default
        assert params.value_size == 32  # Default

    def test_entry_size(self):
        params = KPIRParams(
            num_items_expected=100,
            key_size=8,
            value_size=24,
        )
        # entry_size = key_size + value_size
        assert params.entry_size == 32
        # num_buckets = num_items_expected * expansion_factor
        assert params.num_buckets == 100 * 3


class TestKeywordPIREndToEnd:
    """End-to-end tests for Keyword PIR."""

    @pytest.fixture
    def small_setup(self):
        """Small setup for fast tests."""
        return create_keyword_setup(num_items=50, security_param=40)

    def test_offline_phase(self, small_setup):
        params, kv_pairs, server, client = small_setup
        client.generate_hints(server.stream_database())
        assert client.remaining_queries() > 0

    def test_single_query(self, small_setup):
        params, kv_pairs, server, client = small_setup
        client.generate_hints(server.stream_database())

        key = b"key_000000000000"
        expected_value = kv_pairs[key]

        queries = client.query([key])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()

        assert result == expected_value

    def test_multiple_queries(self, small_setup):
        params, kv_pairs, server, client = small_setup
        client.generate_hints(server.stream_database())

        # Query different keys one at a time
        keys_to_query = [
            b"key_000000000000",
            b"key_000000000010",
            b"key_000000000025",
            b"key_000000000049",
        ]

        for key in keys_to_query:
            if client.remaining_queries() == 0:
                break
            queries = client.query([key])
            responses, stash = server.answer(queries)
            [result] = client.extract(responses, stash)
            client.replenish_hints()
            assert result == kv_pairs[key], f"Mismatch for key {key}"

    def test_batch_query(self, small_setup):
        params, kv_pairs, server, client = small_setup
        client.generate_hints(server.stream_database())

        # Query multiple keys in one batch
        keys = [
            b"key_000000000005",
            b"key_000000000015",
        ]

        queries = client.query(keys)
        responses, stash = server.answer(queries)
        results = client.extract(responses, stash)
        client.replenish_hints()

        for key, result in zip(keys, results):
            assert result == kv_pairs[key], f"Mismatch for key {key}"

    def test_query_nonexistent_key(self, small_setup):
        params, kv_pairs, server, client = small_setup
        client.generate_hints(server.stream_database())

        # Query a key that doesn't exist
        key = b"nonexistent_key!"  # 16 bytes

        queries = client.query([key])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()

        assert result is None

    def test_mixed_existing_nonexisting(self, small_setup):
        params, kv_pairs, server, client = small_setup
        client.generate_hints(server.stream_database())

        keys = [
            b"key_000000000000",  # exists
            b"nonexistent_key!",  # doesn't exist (16 bytes)
            b"key_000000000020",  # exists
        ]

        queries = client.query(keys)
        responses, stash = server.answer(queries)
        results = client.extract(responses, stash)
        client.replenish_hints()

        assert results[0] == kv_pairs[keys[0]]
        assert results[1] is None
        assert results[2] == kv_pairs[keys[2]]


class TestKeywordUpdates:
    """Test database update functionality."""

    @pytest.fixture
    def setup(self):
        params, kv_pairs, server, client = create_keyword_setup(num_items=50, security_param=40)
        client.generate_hints(server.stream_database())
        return params, kv_pairs, server, client

    def test_update_single_key(self, setup):
        params, kv_pairs, server, client = setup

        key = b"key_000000000010"
        new_value = b"updated_value_______________!!!!"  # 32 bytes

        updates = server.update({key: new_value})
        client.update_hints(updates)

        queries = client.query([key])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()

        assert result == new_value

    def test_insert_new_key(self, setup):
        params, kv_pairs, server, client = setup

        key = b"new_key_________"  # 16 bytes
        value = b"new_value_______________________"  # 32 bytes

        updates = server.update({key: value})
        client.update_hints(updates)

        queries = client.query([key])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()

        assert result == value

    def test_delete_key(self, setup):
        params, kv_pairs, server, client = setup

        key = b"key_000000000010"

        updates = server.update({key: None})
        client.update_hints(updates)

        queries = client.query([key])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()

        assert result is None

    def test_delete_nonexistent_key_raises(self, setup):
        params, kv_pairs, server, client = setup

        key = b"nonexistent_key!"  # 16 bytes

        with pytest.raises(KeyError):
            server.update({key: None})

    def test_update_doesnt_break_other_keys(self, setup):
        params, kv_pairs, server, client = setup

        # Update one key
        updates = server.update({
            b"key_000000000010": b"updated_value_______________!!!!"
        })
        client.update_hints(updates)

        # Query other keys - should still work
        other_keys = [
            b"key_000000000000",
            b"key_000000000025",
            b"key_000000000049",
        ]

        for key in other_keys:
            queries = client.query([key])
            responses, stash = server.answer(queries)
            [result] = client.extract(responses, stash)
            client.replenish_hints()
            assert result == kv_pairs[key], f"Mismatch for unmodified key {key}"

    def test_batch_insert_update_delete(self, setup):
        """Test multiple inserts, updates, and deletes in one call."""
        params, kv_pairs, server, client = setup

        # Batch: insert new key, update existing key, delete existing key
        updates = server.update({
            b"new_key_________": b"new_value_______________________",  # insert
            b"key_000000000005": b"updated_value___________________",  # update
            b"key_000000000010": None,  # delete
        })
        client.update_hints(updates)

        # Verify insert
        queries = client.query([b"new_key_________"])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()
        assert result == b"new_value_______________________"

        # Verify update
        queries = client.query([b"key_000000000005"])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()
        assert result == b"updated_value___________________"

        # Verify delete
        queries = client.query([b"key_000000000010"])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()
        assert result is None


class TestKeywordStashOperations:
    """Test operations involving stash."""

    def test_insert_to_stash(self):
        """Insert that goes to stash should still be queryable."""
        # Small table to force stash usage
        params, kv_pairs, server, client = create_keyword_setup(
            num_items=10,
            security_param=40,
            expansion_factor=1,  # Very tight, will cause stash
        )
        client.generate_hints(server.stream_database())

        # Insert more keys - some will go to stash
        for i in range(10, 15):
            key = f"key_{i:012d}".encode()
            value = f"value_{i:026d}".encode()
            updates = server.update({key: value})
            client.update_hints(updates)

        # Query all keys including new ones
        for i in range(15):
            key = f"key_{i:012d}".encode()
            expected = f"value_{i:026d}".encode()
            queries = client.query([key])
            responses, stash = server.answer(queries)
            [result] = client.extract(responses, stash)
            client.replenish_hints()
            assert result == expected, f"Key {i} not found or wrong value"

    def test_delete_from_stash_via_update(self):
        """Delete a key that's in stash."""
        kw_params = KPIRParams(
            num_items_expected=10,
            key_size=16,
            value_size=32,
            expansion_factor=1,
        )
        kv_pairs = {
            f"key_{i:012d}".encode(): f"value_{i:026d}".encode()
            for i in range(15)  # More than buckets, some go to stash
        }

        server = KPIRServer.create(kv_pairs, kw_params, RMS24Server, 40)
        client = KPIRClient.create(kw_params, RMS24Client, 40)
        client.generate_hints(server.stream_database())

        # Find a key in stash
        _, stash = server.answer([])
        if len(stash) == 0:
            pytest.skip("No items in stash for this test")

        stash_key, _ = stash[0]

        # Delete it
        updates = server.update({stash_key: None})
        client.update_hints(updates)

        # Verify it's gone
        queries = client.query([stash_key])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()
        assert result is None


class TestKeywordPIRLarger:
    """Test with larger databases."""

    def test_larger_database(self):
        """Test with 500 items."""
        kw_params = KPIRParams(
            num_items_expected=500,
            key_size=16,
            value_size=64,
        )

        kv_pairs = {}
        for i in range(500):
            key = f"key_{i:012d}".encode()
            value = secrets.token_bytes(64)
            kv_pairs[key] = value

        server = KPIRServer.create(kv_pairs, kw_params, RMS24Server, 40)
        client = KPIRClient.create(kw_params, RMS24Client, 40)
        client.generate_hints(server.stream_database())

        # Query random keys
        keys_list = list(kv_pairs.keys())
        for _ in range(min(10, client.remaining_queries())):
            key = secrets.choice(keys_list)
            queries = client.query([key])
            responses, stash = server.answer(queries)
            [result] = client.extract(responses, stash)
            client.replenish_hints()
            assert result == kv_pairs[key]


class TestRemainingQueries:
    """Test remaining_queries tracking."""

    def test_remaining_queries_decreases(self):
        params, kv_pairs, server, client = create_keyword_setup(
            num_items=30,
            security_param=40,
        )
        client.generate_hints(server.stream_database())

        initial = client.remaining_queries()

        # Each keyword query consumes kappa backup hints
        queries = client.query([b"key_000000000000"])
        responses, stash = server.answer(queries)
        client.extract(responses, stash)
        client.replenish_hints()

        # Should decrease by 1 (for keyword queries)
        assert client.remaining_queries() == initial - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
