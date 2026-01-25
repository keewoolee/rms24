"""
Server implementation for RMS24 single-server PIR.

The server's role is simple:
1. Store the database
2. Stream the database to clients during offline phase
3. Answer online queries by computing XOR parities of requested subsets
"""

from typing import Iterator

from .params import Params
from .messages import Query, Response, EntryUpdate
from .utils import xor_bytes, zero_entry


class Server:
    """
    PIR Server for the single-server RMS24 scheme.

    The server holds the database and answers queries without
    learning which entry the client is interested in.
    """

    def __init__(self, database: list[bytes], params: Params):
        """
        Initialize server with database.

        Args:
            database: List of database entries
            params: PIR parameters
        """
        # Pad database to full size (num_blocks * block_size)
        full_size = params.num_blocks * params.block_size
        self._database = database + [bytes(params.entry_size)] * (full_size - len(database))
        self.params = params

    def stream_database(self) -> Iterator[list[bytes]]:
        """
        Stream the database block by block.

        Used during the client's offline phase.

        Yields:
            Entries for each block, in order
        """
        block_size = self.params.block_size
        for block in range(self.params.num_blocks):
            start = block * block_size
            yield self._database[start:start + block_size]

    def answer(self, queries: list[Query]) -> list[Response]:
        """
        Answer multiple queries by computing parities of subsets.

        Args:
            queries: List of queries containing mask and offsets

        Returns:
            List of responses containing XOR parities of each subset
        """
        num_blocks = self.params.num_blocks
        block_size = self.params.block_size

        responses = []
        for query in queries:
            mask_int = int.from_bytes(query.mask, "little")

            parity_0 = zero_entry(self.params.entry_size)
            parity_1 = zero_entry(self.params.entry_size)
            idx_0 = idx_1 = 0

            for block in range(num_blocks):
                if mask_int & 1:
                    parity_0 = xor_bytes(parity_0, self._database[block * block_size + query.offsets[idx_0]])
                    idx_0 += 1
                else:
                    parity_1 = xor_bytes(parity_1, self._database[block * block_size + query.offsets[idx_1]])
                    idx_1 += 1
                mask_int >>= 1

            responses.append(Response(parity_0=parity_0, parity_1=parity_1))

        return responses

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
            old_value = self._database[index]
            delta = xor_bytes(old_value, new_value)
            self._database[index] = new_value
            result.append(EntryUpdate(index=index, delta=delta))
        return result
