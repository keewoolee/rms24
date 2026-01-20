"""
Tests for Keyword PIR implementation.
"""

import pytest
import secrets
import sys

sys.path.insert(0, "..")

from rms24.keyword_pir import KeywordParams, KeywordClient, KeywordServer


class TestKeywordParams:
    """Test Keyword PIR parameter computation."""

    def test_basic_params(self):
        params = KeywordParams(
            num_items=1000,
            key_size=16,
            value_size=32,
        )
        assert params.num_items == 1000
        assert params.key_size == 16
        assert params.value_size == 32
        assert params.cuckoo_params.num_hashes == 2  # Default κ
        assert params.cuckoo_params.num_buckets >= 1000

    def test_default_sizes(self):
        params = KeywordParams(num_items=100)
        assert params.key_size == 32  # Default
        assert params.value_size == 32  # Default

    def test_pir_params_derived(self):
        params = KeywordParams(
            num_items=100,
            key_size=8,
            value_size=24,
        )
        # PIR entry_size = key_size + value_size
        assert params.pir_params.entry_size == 32
        # PIR n = num_buckets
        assert params.pir_params.n == params.cuckoo_params.num_buckets


class TestKeywordPIREndToEnd:
    """End-to-end tests for Keyword PIR."""

    @pytest.fixture
    def small_setup(self):
        """Small setup for fast tests."""
        params = KeywordParams(
            num_items=50,
            key_size=16,
            value_size=32,
            lambda_=20,
        )
        # Create key-value pairs
        kv_pairs = {}
        for i in range(50):
            key = f"key_{i:012d}".encode()
            value = f"value_{i:026d}".encode()
            kv_pairs[key] = value

        server = KeywordServer(kv_pairs, params)
        client = KeywordClient(params, server.cuckoo_seed)
        return params, kv_pairs, server, client

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
        params = KeywordParams(
            num_items=50,
            key_size=16,
            value_size=32,
            lambda_=20,
        )
        kv_pairs = {}
        for i in range(50):
            key = f"key_{i:012d}".encode()
            value = f"value_{i:026d}".encode()
            kv_pairs[key] = value

        server = KeywordServer(kv_pairs, params)
        client = KeywordClient(params, server.cuckoo_seed)
        client.generate_hints(server.stream_database())
        return params, kv_pairs, server, client

    def test_update_single_key(self, setup):
        params, kv_pairs, server, client = setup

        key = b"key_000000000010"
        new_value = b"updated_value_______________!!!!"  # 32 bytes

        updates = server.update_entries({key: new_value})
        client.update_hints(updates)

        queries = client.query([key])
        responses, stash = server.answer(queries)
        [result] = client.extract(responses, stash)
        client.replenish_hints()

        assert result == new_value

    def test_update_nonexistent_key_raises(self, setup):
        params, kv_pairs, server, client = setup

        key = b"nonexistent_key!"  # 16 bytes
        new_value = b"some_value______________________"  # 32 bytes

        with pytest.raises(KeyError):
            server.update_entries({key: new_value})

    def test_update_doesnt_break_other_keys(self, setup):
        params, kv_pairs, server, client = setup

        # Update one key
        updates = server.update_entries({
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


class TestKeywordPIRLarger:
    """Test with larger databases."""

    def test_larger_database(self):
        """Test with 500 items."""
        params = KeywordParams(
            num_items=500,
            key_size=16,
            value_size=64,
            lambda_=20,
        )

        kv_pairs = {}
        for i in range(500):
            key = f"key_{i:012d}".encode()
            value = secrets.token_bytes(64)
            kv_pairs[key] = value

        server = KeywordServer(kv_pairs, params)
        client = KeywordClient(params, server.cuckoo_seed)
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
        params = KeywordParams(
            num_items=30,
            key_size=16,
            value_size=32,
            lambda_=10,
        )
        kv_pairs = {
            f"key_{i:012d}".encode(): f"value_{i:026d}".encode()
            for i in range(30)
        }

        server = KeywordServer(kv_pairs, params)
        client = KeywordClient(params, server.cuckoo_seed)
        client.generate_hints(server.stream_database())

        initial = client.remaining_queries()

        # Each keyword query consumes κ backup hints
        queries = client.query([b"key_000000000000"])
        responses, stash = server.answer(queries)
        client.extract(responses, stash)
        client.replenish_hints()

        # Should decrease by 1 (for keyword queries)
        assert client.remaining_queries() == initial - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
