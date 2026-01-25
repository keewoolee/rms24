"""
Message types for RMS24 PIR scheme.
"""

from dataclasses import dataclass


@dataclass
class Query:
    """
    Query from client to server.

    The mask indicates which blocks belong to subset_0 (bit k = 1 means
    block k is in subset_0). Both subsets share the same offsets, paired
    by sorted block position. The client hides which subset contains the
    real query.
    """

    mask: bytes  # num_blocks bits: bit k = 1 means block k in subset_0
    offsets: list[int]  # num_blocks/2 offsets, shared by both subsets


@dataclass
class Response:
    """
    Response from server to client.

    Contains XOR parities of entries in each subset.
    """

    parity_0: bytes  # XOR of entries in subset_0
    parity_1: bytes  # XOR of entries in subset_1


@dataclass
class EntryUpdate:
    """A single database entry update."""

    index: int
    delta: bytes  # old_value XOR new_value
