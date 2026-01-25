"""
KPIR Parameters (Section 5, eprint 2019/1483).

Contains cuckoo hashing configuration. The underlying PIR parameters
are managed separately by the specific PIR scheme being used.
"""

from dataclasses import dataclass
from typing import Optional

from .cuckoo import CuckooParams


@dataclass
class KPIRParams:
    """KPIR parameters wrapping cuckoo hashing configuration."""

    num_items_expected: int  # Expected number of key-value pairs (for sizing)
    key_size: int = 32  # Size of keys in bytes
    value_size: int = 32  # Size of values in bytes
    num_hashes: int = 2  # Cuckoo hash functions
    expansion_factor: int = 3  # num_buckets = expansion_factor * num_items_expected
    max_evictions: int = 100  # Max eviction chain length before using stash
    cuckoo_seed: Optional[bytes] = None  # Hash seed (generated if not provided)

    def __post_init__(self):
        if self.num_items_expected < 1:
            raise ValueError("num_items_expected must be at least 1")
        if self.expansion_factor < 1:
            raise ValueError("expansion_factor must be at least 1")

        # Compute number of buckets from expansion factor
        num_buckets = self.num_items_expected * self.expansion_factor

        # Create cuckoo parameters
        self.cuckoo_params = CuckooParams(
            num_buckets=num_buckets,
            key_size=self.key_size,
            value_size=self.value_size,
            num_hashes=self.num_hashes,
            max_evictions=self.max_evictions,
            seed=self.cuckoo_seed,
        )
        # Sync back (in case seed was generated)
        self.cuckoo_seed = self.cuckoo_params.seed

    @property
    def num_buckets(self) -> int:
        """Number of cuckoo buckets (= PIR database size)."""
        return self.cuckoo_params.num_buckets

    @property
    def entry_size(self) -> int:
        """Size of each entry (key_size + value_size)."""
        return self.cuckoo_params.entry_size
