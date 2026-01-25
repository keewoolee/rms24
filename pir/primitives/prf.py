"""
Pseudorandom Function (PRF) Protocol.

A PRF is a keyed function that produces pseudorandom output from any input.
Can be instantiated with various constructions (e.g., AES, ChaCha20, SHA256).
"""

from typing import Protocol


class PRFProtocol(Protocol):
    """
    Generic PRF interface.

    A PRF takes arbitrary input and produces pseudorandom output.
    """

    @property
    def key(self) -> bytes:
        """Get the PRF secret key."""
        ...

    def evaluate(self, input: bytes) -> bytes:
        """
        Evaluate PRF on input.

        Args:
            input: Input bytes

        Returns:
            Pseudorandom output bytes
        """
        ...


