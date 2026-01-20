"""
Correctness tests for RMS24 PIR implementation.
"""

import pytest
import secrets
import sys
sys.path.insert(0, "..")

from rms24.params import Params
from rms24.prf import PRF, find_median_cutoff
from rms24.client import Client
from rms24.server import Server
from rms24.utils import (
    xor_bytes,
    create_sequential_database,
    create_random_database,
)


class TestParams:
    """Test parameter computation."""

    def test_basic_params(self):
        params = Params(n=1024, entry_size=32)
        assert params.n == 1024
        assert params.w == 32  # default: sqrt(1024) = 32
        assert params.c == 32  # n/w = 1024/32 = 32

    def test_non_power_of_two(self):
        """n doesn't have to be a power of two."""
        params = Params(n=1000, entry_size=32)
        assert params.n == 1000
        # ceil(sqrt(1000)) = 32, which is even
        assert params.w == 32
        assert params.c == 32  # ceil(1000/32) = 32

    def test_c_is_even(self):
        """c must be even (hints select c/2 blocks)."""
        # sqrt(100) = 10, w=10, c=ceil(100/10)=10 (even)
        params = Params(n=100, entry_size=32)
        assert params.w == 10
        assert params.c == 10
        assert params.c % 2 == 0

        # sqrt(50) = 7.07, ceil = 8, w=8, c=ceil(50/8)=7 -> 8 (rounded to even)
        params = Params(n=50, entry_size=32)
        assert params.w == 8
        assert params.c == 8  # was 7, rounded up to even
        assert params.c % 2 == 0

        # sqrt(80) = 8.94, ceil = 9, w=9, c=ceil(80/9)=9 -> 10 (rounded to even)
        params = Params(n=80, entry_size=32)
        assert params.w == 9
        assert params.c == 10  # was 9, rounded up to even
        assert params.c % 2 == 0

    def test_block_calculation(self):
        params = Params(n=1024, entry_size=32)
        # Index 0 is in block 0
        assert params.block_of(0) == 0
        # Index 31 is in block 0
        assert params.block_of(31) == 0
        # Index 32 is in block 1
        assert params.block_of(32) == 1
        # Last index
        assert params.block_of(1023) == 31

    def test_offset_calculation(self):
        params = Params(n=1024, entry_size=32)
        assert params.offset_in_block(0) == 0
        assert params.offset_in_block(31) == 31
        assert params.offset_in_block(32) == 0
        assert params.offset_in_block(33) == 1

    def test_index_from_block_offset(self):
        params = Params(n=1024, entry_size=32)
        assert params.index_from_block_offset(0, 0) == 0
        assert params.index_from_block_offset(0, 31) == 31
        assert params.index_from_block_offset(1, 0) == 32
        assert params.index_from_block_offset(31, 31) == 1023


class TestPRF:
    """Test PRF implementation."""

    def test_deterministic(self):
        prf = PRF(b"0" * 16)
        v1 = prf.select(0, 0)
        v2 = prf.select(0, 0)
        assert v1 == v2

    def test_different_inputs(self):
        prf = PRF(b"0" * 16)
        v1 = prf.select(0, 0)
        v2 = prf.select(0, 1)
        v3 = prf.select(1, 0)
        # Should be different with overwhelming probability
        assert len({v1, v2, v3}) == 3

    def test_median_cutoff(self):
        values = [10, 20, 30, 40, 50, 60]
        cutoff, unselected = find_median_cutoff(values)
        below = sum(1 for v in values if v < cutoff)
        at_or_above = sum(1 for v in values if v >= cutoff)
        assert below == 3
        assert at_or_above == 3
        assert len(unselected) == 3
        assert all(values[i] >= cutoff for i in unselected)

    def test_median_cutoff_collision(self):
        """When middle values collide, cutoff should be None."""
        values = [10, 20, 30, 30, 50, 60]  # Collision at median
        cutoff, unselected = find_median_cutoff(values)
        assert cutoff is None
        assert unselected == []


class TestXOR:
    """Test XOR utilities."""

    def test_xor_bytes(self):
        a = bytes([0xFF, 0x00, 0xAA])
        b = bytes([0x0F, 0xF0, 0x55])
        result = xor_bytes(a, b)
        assert result == bytes([0xF0, 0xF0, 0xFF])

    def test_xor_inverse(self):
        a = secrets.token_bytes(32)
        b = secrets.token_bytes(32)
        # (a XOR b) XOR b = a
        result = xor_bytes(xor_bytes(a, b), b)
        assert result == a


class TestEndToEnd:
    """End-to-end correctness tests."""

    @pytest.fixture
    def small_setup(self):
        """Small setup for fast tests.

        Note: lambda_=20 ensures negligible failure probability.
        With w=16 and M=320 hints, P(no hint contains index) < 10^-4
        """
        params = Params(n=256, entry_size=16, lambda_=20)
        db = create_sequential_database(params.n, params.entry_size)
        server = Server(db, params)
        client = Client(params)
        return params, db, server, client

    def test_offline_phase(self, small_setup):
        params, db, server, client = small_setup
        client.generate_hints(server.stream_database())
        assert client.remaining_queries() > 0

    def test_single_query(self, small_setup):
        params, db, server, client = small_setup
        client.generate_hints(server.stream_database())

        # Query index 0
        queries = client.query([0])
        responses = server.answer(queries)
        [result] = client.extract(responses)
        client.replenish_hints()
        assert result == db[0]

    def test_multiple_queries(self, small_setup):
        params, db, server, client = small_setup
        client.generate_hints(server.stream_database())

        # Query multiple indices one at a time
        test_indices = [0, 1, 10, 50, 100, params.n - 1]
        for idx in test_indices:
            if client.remaining_queries() == 0:
                break
            queries = client.query([idx])
            responses = server.answer(queries)
            [result] = client.extract(responses)
            client.replenish_hints()
            assert result == db[idx], f"Mismatch at index {idx}"

    def test_random_queries(self, small_setup):
        params, db, server, client = small_setup
        client.generate_hints(server.stream_database())

        # Query random indices
        num_queries = min(20, client.remaining_queries())
        for _ in range(num_queries):
            idx = secrets.randbelow(params.n)
            queries = client.query([idx])
            responses = server.answer(queries)
            [result] = client.extract(responses)
            client.replenish_hints()
            assert result == db[idx], f"Mismatch at index {idx}"

    def test_repeated_queries_same_index(self, small_setup):
        params, db, server, client = small_setup
        client.generate_hints(server.stream_database())

        # Query the same index multiple times
        idx = 42
        for _ in range(min(5, client.remaining_queries())):
            queries = client.query([idx])
            responses = server.answer(queries)
            [result] = client.extract(responses)
            client.replenish_hints()
            assert result == db[idx]

    def test_random_database(self):
        """Test with random (non-sequential) database."""
        params = Params(n=256, entry_size=32, lambda_=20)
        db = create_random_database(params.n, params.entry_size)
        server = Server(db, params)
        client = Client(params)

        client.generate_hints(server.stream_database())

        # Query random indices
        for _ in range(min(10, client.remaining_queries())):
            idx = secrets.randbelow(params.n)
            queries = client.query([idx])
            responses = server.answer(queries)
            [result] = client.extract(responses)
            client.replenish_hints()
            assert result == db[idx]

    def test_batch_query(self, small_setup):
        """Test querying multiple indices in a single batch."""
        params, db, server, client = small_setup
        client.generate_hints(server.stream_database())

        # Query multiple indices in one batch
        indices = [0, 10, 50, 100]
        queries = client.query(indices)
        responses = server.answer(queries)
        results = client.extract(responses)
        client.replenish_hints()

        for idx, result in zip(indices, results):
            assert result == db[idx], f"Mismatch at index {idx}"


class TestHintReplenishment:
    """Test hint replenishment (Algorithm 5)."""

    def test_queries_exhaust_hints(self):
        params = Params(n=64, entry_size=8, lambda_=10)
        db = create_sequential_database(params.n, params.entry_size)
        server = Server(db, params)
        client = Client(params)

        client.generate_hints(server.stream_database())
        initial_remaining = client.remaining_queries()

        # Each query should decrease remaining by 1
        queries = client.query([0])
        responses = server.answer(queries)
        client.extract(responses)
        client.replenish_hints()
        assert client.remaining_queries() == initial_remaining - 1

        queries = client.query([1])
        responses = server.answer(queries)
        client.extract(responses)
        client.replenish_hints()
        assert client.remaining_queries() == initial_remaining - 2

    def test_hint_contains_queried_index(self):
        """After replenishment, new hint should contain queried index."""
        params = Params(n=256, entry_size=16, lambda_=20)
        db = create_sequential_database(params.n, params.entry_size)
        server = Server(db, params)
        client = Client(params)

        client.generate_hints(server.stream_database())

        # Query an index
        idx = 42
        queries = client.query([idx])
        responses = server.answer(queries)
        client.extract(responses)
        client.replenish_hints()

        # The replenished hint should contain idx
        # We can verify by querying it again
        queries = client.query([idx])
        responses = server.answer(queries)
        [result] = client.extract(responses)
        client.replenish_hints()
        assert result == db[idx]


class TestUpdates:
    """Test database update functionality."""

    @pytest.fixture
    def setup(self):
        """Setup for update tests."""
        params = Params(n=256, entry_size=16, lambda_=20)
        db = create_sequential_database(params.n, params.entry_size)
        server = Server(db, params)
        client = Client(params)
        client.generate_hints(server.stream_database())
        return params, db, server, client

    def test_query_after_update(self, setup):
        """Query returns updated value after update."""
        params, db, server, client = setup

        idx = 42
        new_value = b"updated value!!!"  # 16 bytes

        updates = server.update_entries({idx: new_value})
        client.update_hints(updates)

        queries = client.query([idx])
        responses = server.answer(queries)
        [result] = client.extract(responses)
        client.replenish_hints()

        assert result == new_value

    def test_multiple_updates_same_entry(self, setup):
        """Multiple updates to the same entry work correctly."""
        params, db, server, client = setup

        idx = 10
        values = [b"first update!!!!", b"second update!!!", b"third update!!!!"]

        for new_value in values:
            updates = server.update_entries({idx: new_value})
            client.update_hints(updates)

        queries = client.query([idx])
        responses = server.answer(queries)
        [result] = client.extract(responses)
        client.replenish_hints()

        assert result == values[-1]

    def test_batch_updates(self, setup):
        """Batch updates to different entries work correctly."""
        params, db, server, client = setup

        update_map = {
            5: b"value for 5!!!!!",
            50: b"value for 50!!!!",
            100: b"value for 100!!!",
        }

        updates = server.update_entries(update_map)
        client.update_hints(updates)

        for idx, expected in update_map.items():
            queries = client.query([idx])
            responses = server.answer(queries)
            [result] = client.extract(responses)
            client.replenish_hints()
            assert result == expected, f"Mismatch at index {idx}"

    def test_update_doesnt_break_unrelated_queries(self, setup):
        """Updates don't affect queries to unmodified entries."""
        params, db, server, client = setup

        # Update entry 10
        updates = server.update_entries({10: b"new value!!!!!!!"})
        client.update_hints(updates)

        # Query unmodified entries
        for idx in [0, 1, 20, 100, params.n - 1]:
            queries = client.query([idx])
            responses = server.answer(queries)
            [result] = client.extract(responses)
            client.replenish_hints()
            assert result == db[idx], f"Mismatch at unmodified index {idx}"

    def test_update_before_generate_hints_raises(self):
        """update_hints() raises if called before generate_hints()."""
        params = Params(n=256, entry_size=16, lambda_=20)
        db = create_sequential_database(params.n, params.entry_size)
        server = Server(db, params)
        client = Client(params)

        updates = server.update_entries({0: b"some new value!!"})
        with pytest.raises(RuntimeError, match="Must call generate_hints"):
            client.update_hints(updates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
