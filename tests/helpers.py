"""
Test helper functions.
"""

import secrets


def create_random_database(n: int, entry_size: int = 32) -> list[bytes]:
    """Create a database with random entries."""
    return [secrets.token_bytes(entry_size) for _ in range(n)]


def create_sequential_database(n: int, entry_size: int = 32) -> list[bytes]:
    """
    Create a database with sequential values (for testing).

    Each entry contains its index as bytes.
    """
    return [
        i.to_bytes(min(entry_size, 8), "little").ljust(entry_size, b"\x00")
        for i in range(n)
    ]
