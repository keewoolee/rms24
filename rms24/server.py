"""
Server implementation for RMS24 single-server PIR.

The server's role is simple:
1. Store the database
2. Answer online queries by computing XOR parities of requested subsets
3. Stream the database to clients during offline phase
"""

from typing import Iterator

from .params import Params
from .protocol import Query, Response
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

    def answer(self, query: Query) -> Response:
        """
        Answer an online query by computing parities of two subsets.

        The client sends a compressed query (mask + shared offsets).
        Server reconstructs subsets and computes XOR of entries.

        Args:
            query: Query containing mask and shared offsets

        Returns:
            Response containing XOR parities of each subset
        """
        c = self.params.c
        w = self.params.w
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

        return Response(parity_0=parity_0, parity_1=parity_1)

    def stream_database(self) -> Iterator[tuple[int, list[bytes]]]:
        """
        Stream the database block by block.

        Used during the client's offline phase.

        Yields:
            Tuples of (block_id, entries_in_block)
        """
        yield from self.db.stream_blocks(self.params.w, self.params.c)
