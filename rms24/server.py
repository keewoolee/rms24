"""
Server implementation for RMS24 single-server PIR.

The server's role is simple:
1. Store the database
2. Answer online queries by computing XOR parities of requested subsets
3. Stream the database to clients during offline phase
"""

from typing import Iterator

from .params import Params
from .protocol import Query, Response, EntryUpdate
from .utils import Database, xor_bytes, zero_entry


class Server:
    """
    PIR Server for the single-server RMS24 scheme.

    The server holds the database and answers queries without
    learning which entry the client is interested in.
    """

    def __init__(self, database: Database, params: Params):
        """
        Initialize server with database.

        Args:
            database: The database to serve
            params: PIR parameters
        """
        self.db = database
        self.params = params

    def answer(self, queries: list[Query]) -> list[Response]:
        """
        Answer multiple queries by computing parities of subsets.

        Args:
            queries: List of queries containing mask and shared offsets

        Returns:
            List of responses containing XOR parities of each subset
        """
        c = self.params.c
        w = self.params.w

        responses = []
        for query in queries:
            mask_int = int.from_bytes(query.mask, "little")

            parity_0 = zero_entry(self.params.entry_size)
            parity_1 = zero_entry(self.params.entry_size)
            idx_0 = idx_1 = 0

            for k in range(c):
                if mask_int & 1:
                    parity_0 = xor_bytes(parity_0, self.db[k * w + query.offsets[idx_0]])
                    idx_0 += 1
                else:
                    parity_1 = xor_bytes(parity_1, self.db[k * w + query.offsets[idx_1]])
                    idx_1 += 1
                mask_int >>= 1

            responses.append(Response(parity_0=parity_0, parity_1=parity_1))

        return responses

    def stream_database(self) -> Iterator[tuple[int, list[bytes]]]:
        """
        Stream the database block by block.

        Used during the client's offline phase.

        Yields:
            Tuples of (block_id, entries_in_block)
        """
        yield from self.db.stream_blocks(self.params.w, self.params.c)

    def update_entries(self, updates: dict[int, bytes]) -> list[EntryUpdate]:
        """
        Update multiple database entries.

        Args:
            updates: Mapping from index to new value

        Returns:
            List of EntryUpdate for client hint updates
        """
        result = []
        for index, new_value in updates.items():
            old_value = self.db[index]
            delta = xor_bytes(old_value, new_value)
            self.db.entries[index] = new_value
            result.append(EntryUpdate(index=index, delta=delta))
        return result
