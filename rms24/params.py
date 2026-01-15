"""
Parameters for the RMS24 PIR scheme.

Key parameters (using Plinko notation for forward compatibility):
- n: Total number of database entries
- w: Block size (entries per block), defaults to ⌈√n⌉
- c: Number of blocks = ⌈n/w⌉ (derived, must be even)
- lambda_: Security parameter (default 80)
- entry_size: Size of each database entry in bytes
- num_reg_hints: Number of regular hints (default λ * w)
- num_backup_hints: Number of backup hints (configurable, default num_reg_hints)

Tradeoffs:
- Query size grows with c (number of blocks)
- num_reg_hints = λ * w by default, so hint storage grows with w
- num_backup_hints determines queries per offline phase
- Default w = √n balances query size and hint storage to O(√n)
"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class Params:
    """Parameters for RMS24 PIR scheme."""

    n: int                      # Number of database entries
    entry_size: int             # Size of each entry in bytes
    lambda_: int = 80           # Security parameter (paper: λ)
    w: Optional[int] = None     # Block size (entries per block)
    num_backup_hints: Optional[int] = None  # Configurable backup hints

    def __post_init__(self):
        # Validate parameters
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.entry_size < 1:
            raise ValueError("entry_size must be at least 1")
        if self.lambda_ < 1:
            raise ValueError("lambda_ must be at least 1")

        # Default w: ceil(sqrt(n))
        if self.w is None:
            self.w = math.ceil(math.sqrt(self.n))

        if self.w < 1:
            raise ValueError("w must be at least 1")

        # Number of blocks c = ceil(n/w)
        self._c = math.ceil(self.n / self.w)

        # c must be even (hints select exactly c/2 blocks)
        if self._c % 2 == 1:
            self._c += 1

        # Number of regular hints (paper: M = λ * w)
        self._num_reg_hints = self.lambda_ * self.w

        # Default num_backup_hints: num_reg_hints
        if self.num_backup_hints is None:
            self.num_backup_hints = self._num_reg_hints

        if self.num_backup_hints < 0:
            raise ValueError("num_backup_hints must be non-negative")

    @property
    def c(self) -> int:
        """Number of blocks (Plinko: c = n/w)."""
        return self._c

    @property
    def num_reg_hints(self) -> int:
        """Number of regular hints (paper: M = λ * w)."""
        return self._num_reg_hints

    def block_of(self, index: int) -> int:
        """Return the block index for a given database index."""
        return index // self.w

    def offset_in_block(self, index: int) -> int:
        """Return the offset within the block for a given index."""
        return index % self.w

    def index_from_block_offset(self, block: int, offset: int) -> int:
        """Compute database index from block and offset."""
        return block * self.w + offset

    def __repr__(self) -> str:
        return (
            f"Params(n={self.n}, w={self.w}, c={self.c}, "
            f"num_reg_hints={self.num_reg_hints}, "
            f"num_backup_hints={self.num_backup_hints}, "
            f"entry_size={self.entry_size}, lambda={self.lambda_})"
        )
