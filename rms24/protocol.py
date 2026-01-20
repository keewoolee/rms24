"""
Protocol messages for RMS24 PIR.

These are the messages exchanged between client and server.
"""

from dataclasses import dataclass


@dataclass
class Query:
    """
    Query from client to server.

    Uses compressed format where both subsets share the same offsets,
    reducing query size by ~4x. This trick was proposed by Vitalik Buterin.

    The mask indicates which blocks belong to subset_0 (bit k = 1 means
    block k is in subset_0). Both subsets are sorted by block ID and
    paired with offsets by position.
    """
    mask: bytes         # c bits: bit k = 1 means block k in subset_0
    offsets: list[int]  # c/2 offsets, shared by both subsets (by sorted position)


@dataclass
class Response:
    """Response from server to client."""
    parity_0: bytes
    parity_1: bytes


@dataclass
class EntryUpdate:
    """A single database entry update."""
    index: int
    delta: bytes  # old_value ⊕ new_value
