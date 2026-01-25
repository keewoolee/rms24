"""
Parameters for the RMS24 PIR scheme.

Based on "Simple and Practical Amortized Sublinear Private Information
Retrieval" (https://eprint.iacr.org/2024/1362).

Key parameters:
- num_entries: Total number of database entries
- block_size: Entries per block, defaults to ⌈√num_entries⌉
- num_blocks: Number of blocks, must be even
- security_param: Security parameter controlling failure probability
- entry_size: Size of each database entry in bytes
- num_reg_hints: Number of regular hints
- num_backup_hints: Number of backup hints (default num_reg_hints)

Tradeoffs:
- Query size grows with num_blocks
- num_reg_hints = security_param * block_size by default, so hint storage grows with block_size
- num_backup_hints determines queries per offline phase
- Default block_size = √num_entries balances query size and hint storage to O(√num_entries)
"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class Params:
    """Parameters for RMS24 PIR scheme."""

    num_entries: int  # Number of database entries
    entry_size: int  # Size of each entry in bytes
    security_param: int = 80  # Query failure probability ≈ 2^{-security_param}
    block_size: Optional[int] = None  # Entries per block
    num_backup_hints: Optional[int] = None  # Configurable backup hints

    def __post_init__(self):
        # Validate parameters
        if self.num_entries < 1:
            raise ValueError("num_entries must be at least 1")
        if self.entry_size < 1:
            raise ValueError("entry_size must be at least 1")
        if self.security_param < 1:
            raise ValueError("security_param must be at least 1")

        # Default block_size: ceil(sqrt(num_entries))
        if self.block_size is None:
            self.block_size = math.ceil(math.sqrt(self.num_entries))

        if self.block_size < 1:
            raise ValueError("block_size must be at least 1")

        # Number of blocks = ceil(num_entries / block_size)
        self._num_blocks = math.ceil(self.num_entries / self.block_size)

        # num_blocks must be even: each hint splits blocks into two equal halves.
        # Query sends num_blocks/2 offsets that could belong to either half,
        # hiding which half contains the real query.
        if self._num_blocks % 2 == 1:
            self._num_blocks += 1

        # Number of regular hints
        self._num_reg_hints = self.security_param * self.block_size

        # Default num_backup_hints: num_reg_hints
        if self.num_backup_hints is None:
            self.num_backup_hints = self._num_reg_hints

        if self.num_backup_hints < 0:
            raise ValueError("num_backup_hints must be non-negative")

    @property
    def num_blocks(self) -> int:
        """Number of blocks. Always even."""
        return self._num_blocks

    @property
    def num_reg_hints(self) -> int:
        """Number of regular hints."""
        return self._num_reg_hints

    def block_of(self, index: int) -> int:
        """Return the block index for a given database index."""
        return index // self.block_size

    def offset_in_block(self, index: int) -> int:
        """Return the offset within the block for a given index."""
        return index % self.block_size

    def index_from_block_offset(self, block: int, offset: int) -> int:
        """Compute database index from block and offset."""
        return block * self.block_size + offset

    def __repr__(self) -> str:
        return (
            f"Params(num_entries={self.num_entries}, "
            f"block_size={self.block_size}, num_blocks={self.num_blocks}, "
            f"num_reg_hints={self.num_reg_hints}, "
            f"num_backup_hints={self.num_backup_hints}, "
            f"entry_size={self.entry_size}, security_param={self.security_param})"
        )
