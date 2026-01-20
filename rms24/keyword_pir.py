"""
Keyword PIR using Cuckoo Hashing.

This module provides a wrapper around the RMS24 PIR scheme that supports
keyword-based lookups instead of index-based lookups. It uses cuckoo hashing
to map sparse keywords to a dense index space.

Based on Section 5 of "Simple and Practical Amortized Sublinear Private
Information Retrieval using Dummy Subsets".
"""

from dataclasses import dataclass
from typing import Iterator, Optional

from .cuckoo import CuckooParams, CuckooHash, CuckooTable
from .params import Params
from .protocol import Query, Response, EntryUpdate
from .client import Client
from .server import Server
from .utils import Database

import math


@dataclass
class _KeywordQueryState:
    """Internal state from query() needed for extract()."""
    keywords: list[bytes]


@dataclass
class KeywordParams:
    """
    Parameters for Keyword PIR.

    Combines cuckoo hashing parameters with RMS24 PIR parameters.
    """

    num_items: int  # D: number of key-value pairs
    value_size: int = 32  # Size of values in bytes
    key_size: int = 32  # Size of keys in bytes
    lambda_: int = 80  # RMS24 security parameter
    expansion_factor: int = 3  # num_buckets = expansion_factor * num_items
    num_hashes: int = 2  # Cuckoo hash functions (κ)
    max_evictions: int = 500  # Max eviction chain length before using stash

    def __post_init__(self):
        # Compute number of buckets from expansion factor
        num_buckets = self.num_items * self.expansion_factor

        # Create cuckoo parameters
        self._cuckoo_params = CuckooParams(
            key_size=self.key_size,
            value_size=self.value_size,
            num_hashes=self.num_hashes,
            num_buckets=num_buckets,
            max_evictions=self.max_evictions,
        )

        # Create RMS24 parameters
        # n = number of cuckoo buckets
        # entry_size = key_size + value_size
        self._pir_params = Params(
            n=self._cuckoo_params.num_buckets,
            entry_size=self._cuckoo_params.entry_size,
            lambda_=self.lambda_,
        )

    @property
    def cuckoo_params(self) -> CuckooParams:
        """Cuckoo hashing parameters."""
        return self._cuckoo_params

    @property
    def pir_params(self) -> Params:
        """Underlying RMS24 PIR parameters."""
        return self._pir_params


class KeywordServer:
    """
    Keyword PIR Server.

    Builds a cuckoo hash table from key-value pairs and wraps an RMS24 server.
    """

    def __init__(
        self,
        kv_pairs: dict[bytes, bytes],
        params: KeywordParams,
        seed: Optional[bytes] = None,
    ):
        """
        Initialize server with key-value pairs.

        Args:
            kv_pairs: Dictionary mapping keys to values
            params: Keyword PIR parameters
            seed: Optional seed for cuckoo hash functions (for reproducibility)
        """
        self.params = params

        # Build cuckoo hash table
        self._cuckoo_table = CuckooTable.build(
            kv_pairs, params.cuckoo_params, seed=seed
        )

        # Convert to dense database
        entries = self._cuckoo_table.to_database()
        self._database = Database(entries=entries)

        # Wrap with RMS24 server
        self._pir_server = Server(self._database, params.pir_params)

    @property
    def cuckoo_seed(self) -> bytes:
        """Get cuckoo hash seed (needed by client)."""
        return self._cuckoo_table.seed

    def stream_database(self) -> Iterator[tuple[int, list[bytes]]]:
        """
        Stream the database for client hint generation.

        Yields:
            Tuples of (block_id, entries_in_block)
        """
        return self._pir_server.stream_database()

    def answer(self, queries: list[Query]) -> tuple[list[Response], list[tuple[bytes, bytes]]]:
        """
        Answer PIR queries.

        Args:
            queries: List of PIR queries

        Returns:
            Tuple of (PIR responses, current stash entries)
        """
        return self._pir_server.answer(queries), self._cuckoo_table.stash

    def update_entries(self, updates: dict[bytes, bytes]) -> list[EntryUpdate]:
        """
        Update key-value pairs.

        Args:
            updates: Dictionary mapping keys to new values

        Returns:
            List of EntryUpdate for client hint updates (only for bucket updates)

        Raises:
            KeyError: If a key doesn't exist in the table
        """
        index_updates = {}
        for key, new_value in updates.items():
            bucket_idx = self._cuckoo_table.update(key, new_value)
            if bucket_idx is not None:
                index_updates[bucket_idx] = key + new_value
            # Stash updates don't need PIR updates - client has stash directly

        return self._pir_server.update_entries(index_updates)


class KeywordClient:
    """
    Keyword PIR Client.

    Wraps an RMS24 client and handles keyword-to-index translation via
    cuckoo hashing.
    """

    def __init__(
        self,
        params: KeywordParams,
        cuckoo_seed: bytes,
    ):
        """
        Initialize client.

        Args:
            params: Keyword PIR parameters
            cuckoo_seed: Cuckoo hash seed from server
        """
        self.params = params
        cuckoo = params.cuckoo_params
        self._hasher = CuckooHash(
            cuckoo.num_hashes, cuckoo.num_buckets, seed=cuckoo_seed
        )
        self._pir_client = Client(params.pir_params)
        self._query_state: Optional[_KeywordQueryState] = None

    def generate_hints(self, db_stream: Iterator[tuple[int, list[bytes]]]) -> None:
        """
        Generate hints from database stream.

        Args:
            db_stream: Iterator from server.stream_database()
        """
        self._pir_client.generate_hints(db_stream)

    def query(self, keywords: list[bytes]) -> list[Query]:
        """
        Prepare queries for keywords.

        Each keyword requires κ PIR queries (one per hash function).

        Args:
            keywords: List of keywords to look up

        Returns:
            List of κ * len(keywords) PIR queries
        """
        if self._query_state is not None:
            raise RuntimeError("Previous query batch not yet completed")

        # Compute all hash positions
        indices = []
        for keyword in keywords:
            positions = self._hasher.all_positions(keyword)
            indices.extend(positions)

        self._query_state = _KeywordQueryState(keywords=keywords)

        return self._pir_client.query(indices)

    def _find_value(
        self,
        keyword: bytes,
        bucket_entries: list[bytes],
        stash_dict: dict[bytes, bytes],
    ) -> Optional[bytes]:
        """Find value for keyword in bucket entries or stash."""
        key_size = self.params.key_size
        for entry in bucket_entries:
            if entry[:key_size] == keyword:
                return entry[key_size:]
        return stash_dict.get(keyword)

    def extract(
        self,
        responses: list[Response],
        stash: list[tuple[bytes, bytes]],
    ) -> list[Optional[bytes]]:
        """
        Extract values from responses.

        Args:
            responses: List of PIR responses
            stash: Stash entries from server

        Returns:
            List of values (None if keyword not found)
        """
        if self._query_state is None:
            raise RuntimeError("Must call query() before extract()")

        entries = self._pir_client.extract(responses)
        kappa = self.params.cuckoo_params.num_hashes
        stash_dict = dict(stash)

        results = []
        for i, keyword in enumerate(self._query_state.keywords):
            bucket_entries = entries[i * kappa : (i + 1) * kappa]
            results.append(self._find_value(keyword, bucket_entries, stash_dict))

        return results

    def replenish_hints(self) -> None:
        """
        Replenish consumed hints.

        Must be called after extract() to complete the query batch.
        """
        self._pir_client.replenish_hints()
        self._query_state = None

    def update_hints(self, updates: list[EntryUpdate]) -> None:
        """
        Update hints for database changes.

        Args:
            updates: List of EntryUpdate from server
        """
        self._pir_client.update_hints(updates)

    def remaining_queries(self) -> int:
        """
        Return number of keyword queries remaining.

        Note: Each keyword query uses κ PIR queries internally,
        so this is remaining_pir_queries // κ.
        """
        return self._pir_client.remaining_queries() // self.params.cuckoo_params.num_hashes
