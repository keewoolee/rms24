"""
PRF and utility functions for RMS24.
"""

import hmac
import hashlib
import secrets
import struct


class HMACPRF:
    """
    HMAC-SHA-256 based PRF for RMS24.

    Security: The PRF key is client-secret and must never be shared with
    the server. The server learns nothing about which entries the client
    accesses as long as the key remains secret.
    """

    def __init__(self, key: bytes = None):
        """
        Initialize PRF with a secret key.

        Args:
            key: 32-byte key. If None, generates a random key.
        """
        if key is None:
            key = secrets.token_bytes(32)
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        self._key = key

    @property
    def key(self) -> bytes:
        """Get the PRF secret key."""
        return self._key

    def evaluate(self, input: bytes) -> bytes:
        """
        Evaluate PRF on input.

        Args:
            input: Input bytes

        Returns:
            32-byte pseudorandom output
        """
        return hmac.new(self._key, input, hashlib.sha256).digest()

    def select(self, hint_id: int, block: int) -> int:
        """Return 32-bit value for block selection."""
        input = b"select" + struct.pack("<II", hint_id, block)
        output = self.evaluate(input)
        return int.from_bytes(output[:4], "little")

    def offset(self, hint_id: int, block: int) -> int:
        """Return 256-bit value for offset selection.

        Uses full 256-bit output to avoid modular bias when computing % block_size.
        """
        input = b"offset" + struct.pack("<II", hint_id, block)
        output = self.evaluate(input)
        return int.from_bytes(output, "little")

    def select_vector(self, hint_id: int, num_blocks: int) -> list[int]:
        """Compute select values for all blocks."""
        return [self.select(hint_id, k) for k in range(num_blocks)]


def find_median_cutoff(values: list[int]) -> int:
    """
    Find the median cutoff value.

    Args:
        values: List of PRF select values (length must be even)

    Returns:
        cutoff such that exactly len(values)/2 elements are smaller,
        or 0 if the two middle values collide (~2^-32 probability)
    """
    if len(values) % 2 != 0:
        raise ValueError("Length must be even")

    # Possible optimization: Full sort is O(n log n). A selection algorithm
    # like quickselect or numpy.partition would be O(n) average.
    sorted_values = sorted(values)
    mid = len(values) // 2

    # Check for collision at median boundary
    if sorted_values[mid - 1] == sorted_values[mid]:
        return 0

    return sorted_values[mid]


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
