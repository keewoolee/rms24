"""
Utility functions for RMS24.
"""

import secrets
from typing import Iterator


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """
    XOR two byte strings of equal length.

    Args:
        a: First byte string
        b: Second byte string

    Returns:
        XOR of a and b
    """
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    return bytes(x ^ y for x, y in zip(a, b))


def zero_entry(entry_size: int) -> bytes:
    """Create a zero-filled entry."""
    return bytes(entry_size)


class Database:
    """
    Simple in-memory database for testing.

    Stores n entries of fixed size.
    """

    def __init__(self, entries: list[bytes] = None, n: int = 0, entry_size: int = 32):
        """
        Initialize database.

        Args:
            entries: List of byte entries (if provided)
            n: Number of entries to create (if entries not provided)
            entry_size: Size of each entry in bytes
        """
        if entries is not None:
            self.entries = entries
            self.entry_size = len(entries[0]) if entries else entry_size
        else:
            # Create random entries
            self.entries = [secrets.token_bytes(entry_size) for _ in range(n)]
            self.entry_size = entry_size

    def __getitem__(self, index: int) -> bytes:
        """Get entry at index."""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        # Return zero for out-of-bounds (for padding)
        return zero_entry(self.entry_size)

    def __len__(self) -> int:
        return len(self.entries)

    def get_block(self, block_id: int, w: int) -> list[bytes]:
        """
        Get all entries in a block.

        Args:
            block_id: Block index k
            w: Number of entries per block (block size)

        Returns:
            List of entries in block k
        """
        start = block_id * w
        end = start + w
        return [self[i] for i in range(start, end)]

    def stream_blocks(self, w: int, c: int) -> Iterator[tuple[int, list[bytes]]]:
        """
        Stream database block by block.

        Args:
            w: Block size (entries per block)
            c: Number of blocks to yield (from params.c)

        Yields:
            Tuples of (block_id, entries_in_block)
        """
        for k in range(c):
            yield k, self.get_block(k, w)


def create_random_database(n: int, entry_size: int = 32) -> Database:
    """Create a database with random entries."""
    return Database(n=n, entry_size=entry_size)


def create_sequential_database(n: int, entry_size: int = 32) -> Database:
    """
    Create a database with sequential values (for testing).

    Each entry contains its index as bytes.
    """
    entries = []
    for i in range(n):
        # Store index as bytes, padded to entry_size
        entry = i.to_bytes(min(entry_size, 8), "little").ljust(entry_size, b"\x00")
        entries.append(entry)
    return Database(entries=entries)
