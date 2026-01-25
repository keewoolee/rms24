"""
KPIR Server (Section 5, eprint 2019/1483).

Wraps any index-based PIR server with cuckoo hashing for keyword lookups.
"""

from typing import Iterator, Optional

from .cuckoo import CuckooTable
from .params import KPIRParams
from ..protocols import PIRServer, Query, Response, EntryUpdate
from ..rms24 import Params as PIRParams


class KPIRServer:
    """KPIR Server. Wraps a PIR server with cuckoo hashing for keyword-based lookups."""

    def __init__(
        self,
        params: KPIRParams,
        cuckoo_table: CuckooTable,
        pir_server: PIRServer,
    ):
        """
        Initialize server with pre-built components.

        Args:
            params: KPIR parameters
            cuckoo_table: Cuckoo hash table (must match pir_server's database)
            pir_server: Underlying PIR server
        """
        self.params = params
        self._cuckoo_table = cuckoo_table
        self._pir_server = pir_server

    @classmethod
    def create(
        cls,
        kv_pairs: dict[bytes, bytes],
        params: KPIRParams,
        pir_server_factory,
        security_param: int = 80,
    ) -> "KPIRServer":
        """
        Create a KPIRServer from key-value pairs.

        Args:
            kv_pairs: Dictionary mapping keys to values
            params: KPIR parameters (includes cuckoo seed)
            pir_server_factory: Callable(database, pir_params) -> PIRServer
            security_param: Security parameter for underlying PIR (default: 80)

        Returns:
            Configured KPIRServer
        """
        pir_params = PIRParams(
            num_entries=params.num_buckets,
            entry_size=params.entry_size,
            security_param=security_param,
        )
        cuckoo_table = CuckooTable.build(kv_pairs, params.cuckoo_params)
        database = cuckoo_table.to_database()
        pir_server = pir_server_factory(database, pir_params)
        return cls(params, cuckoo_table, pir_server)

    def stream_database(self) -> Iterator[list[bytes]]:
        """
        Stream the database for client hint generation.

        Yields:
            Entries for each block, in order
        """
        return self._pir_server.stream_database()

    def answer(
        self, queries: list[Query]
    ) -> tuple[list[Response], list[tuple[bytes, bytes]]]:
        """
        Answer PIR queries.

        Args:
            queries: List of PIR queries

        Returns:
            Tuple of (PIR responses, current stash entries)
        """
        return self._pir_server.answer(queries), self._cuckoo_table.stash

    def update(self, changes: dict[bytes, Optional[bytes]]) -> list[EntryUpdate]:
        """
        Update the database.

        Args:
            changes: Dictionary mapping keys to values.
                - key -> value: insert (if new) or update (if exists)
                - key -> None: delete

        Returns:
            List of EntryUpdate for client hint updates

        Raises:
            KeyError: If deleting a key that doesn't exist
        """
        index_updates: dict[int, bytes] = {}
        empty_entry = bytes(self.params.entry_size)

        for key, value in changes.items():
            if value is None:
                # Delete
                bucket_idx = self._cuckoo_table.delete(key)
                if bucket_idx is not None:
                    index_updates[bucket_idx] = empty_entry
            else:
                # Try update first, then insert if not found
                try:
                    bucket_idx = self._cuckoo_table.update(key, value)
                    if bucket_idx is not None:
                        index_updates[bucket_idx] = key + value
                except KeyError:
                    # Key doesn't exist, insert
                    bucket_changes = self._cuckoo_table.insert(key, value)
                    for idx, entry in bucket_changes:
                        index_updates[idx] = entry

        return self._pir_server.update_entries(index_updates)
