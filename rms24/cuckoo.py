"""
Cuckoo hashing for Keyword PIR.

Cuckoo hashing maps D key-value pairs into n buckets using κ hash functions,
where each bucket contains at most one item. This enables converting sparse
keyword queries to dense index PIR.
"""

import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional


@dataclass
class CuckooParams:
    """Parameters for cuckoo hash table."""

    key_size: int  # Size of keys in bytes
    value_size: int  # Size of values in bytes
    num_hashes: int  # κ: number of hash functions
    num_buckets: int  # n: number of buckets
    max_evictions: int = 500  # Max eviction chain length before using stash

    def __post_init__(self):
        if self.key_size < 2:
            raise ValueError("key_size must be at least 2")
        if self.value_size < 1:
            raise ValueError("value_size must be at least 1")
        if self.num_hashes < 2:
            raise ValueError("num_hashes must be at least 2")
        if self.num_buckets < 1:
            raise ValueError("num_buckets must be at least 1")

    @property
    def entry_size(self) -> int:
        """Size of each bucket (key || value)."""
        return self.key_size + self.value_size


class CuckooHash:
    """
    Cuckoo hash functions.

    Uses SHA256 with different prefixes to create κ independent hash functions.
    """

    def __init__(self, num_hashes: int, num_buckets: int, seed: Optional[bytes] = None):
        """
        Initialize cuckoo hash functions.

        Args:
            num_hashes: Number of hash functions (κ)
            num_buckets: Number of buckets (n)
            seed: Optional seed for reproducibility
        """
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.seed = seed if seed is not None else secrets.token_bytes(16)

    def hash(self, hash_idx: int, key: bytes) -> int:
        """
        Compute hash function H_i(key).

        Args:
            hash_idx: Which hash function (0 to κ-1)
            key: The key to hash

        Returns:
            Bucket index in [0, num_buckets)
        """
        # Use SHA256 with prefix: seed || hash_idx || key
        h = hashlib.sha256()
        h.update(self.seed)
        h.update(hash_idx.to_bytes(4, "little"))
        h.update(key)
        digest = h.digest()
        # Convert first 8 bytes to int and mod by num_buckets
        value = int.from_bytes(digest[:8], "little")
        return value % self.num_buckets

    def all_positions(self, key: bytes) -> list[int]:
        """
        Get all possible positions for a key.

        Args:
            key: The key to hash

        Returns:
            List of κ bucket indices
        """
        return [self.hash(i, key) for i in range(self.num_hashes)]


class CuckooTable:
    """
    Cuckoo hash table storing (key, value) pairs.

    Each bucket contains at most one item. Empty buckets are all zeros.
    Items that fail insertion go to a stash (overflow area).
    """

    def __init__(self, params: CuckooParams, seed: Optional[bytes] = None):
        """
        Initialize empty cuckoo table.

        Args:
            params: Cuckoo parameters
            seed: Optional seed for hash functions
        """
        self.params = params
        self.hasher = CuckooHash(params.num_hashes, params.num_buckets, seed)

        # Each bucket is (key, value) or None
        self._buckets: list[Optional[tuple[bytes, bytes]]] = [None] * params.num_buckets
        self._stash: list[tuple[bytes, bytes]] = []

    @property
    def seed(self) -> bytes:
        """Get the hash seed (needed by client)."""
        return self.hasher.seed

    def insert(self, key: bytes, value: bytes) -> None:
        """
        Insert a key-value pair.

        If insertion fails after max_evictions, the item goes to the stash.
        Caller must ensure keys are unique; inserting duplicate keys causes
        undefined behavior.

        Args:
            key: Key (must be exactly key_size bytes)
            value: Value (must be exactly value_size bytes)

        Raises:
            ValueError: If key or value has wrong size
        """
        if len(key) != self.params.key_size:
            raise ValueError(
                f"Key size mismatch: {len(key)} != {self.params.key_size}"
            )
        if len(value) != self.params.value_size:
            raise ValueError(
                f"Value size mismatch: {len(value)} != {self.params.value_size}"
            )

        # Try to insert with eviction chain
        current_key, current_value = key, value

        for _ in range(self.params.max_evictions):
            positions = self.hasher.all_positions(current_key)

            # Try to find an empty bucket
            for pos in positions:
                if self._buckets[pos] is None:
                    self._buckets[pos] = (current_key, current_value)
                    return

            # No empty bucket, evict from random position
            evict_pos = secrets.choice(positions)
            evicted = self._buckets[evict_pos]
            self._buckets[evict_pos] = (current_key, current_value)
            current_key, current_value = evicted

        # Eviction chain too long, add to stash
        self._stash.append((current_key, current_value))

    def lookup(self, key: bytes) -> Optional[bytes]:
        """
        Look up a key.

        Args:
            key: Key to look up

        Returns:
            Value if found, None otherwise
        """
        # Check hash positions
        positions = self.hasher.all_positions(key)
        for pos in positions:
            bucket = self._buckets[pos]
            if bucket is not None and bucket[0] == key:
                return bucket[1]

        # Check stash
        for stash_key, stash_value in self._stash:
            if stash_key == key:
                return stash_value

        return None

    def update(self, key: bytes, new_value: bytes) -> Optional[int]:
        """
        Update the value for an existing key.

        Args:
            key: Key to update (must be exactly key_size bytes)
            new_value: New value (must be exactly value_size bytes)

        Returns:
            Bucket index if key was in a bucket, None if key was in stash

        Raises:
            ValueError: If key or new_value has wrong size
            KeyError: If key not found
        """
        if len(key) != self.params.key_size:
            raise ValueError(
                f"Key size mismatch: {len(key)} != {self.params.key_size}"
            )
        if len(new_value) != self.params.value_size:
            raise ValueError(
                f"Value size mismatch: {len(new_value)} != {self.params.value_size}"
            )

        # Check buckets
        positions = self.hasher.all_positions(key)
        for pos in positions:
            bucket = self._buckets[pos]
            if bucket is not None and bucket[0] == key:
                self._buckets[pos] = (key, new_value)
                return pos

        # Check stash
        for i, (stash_key, _) in enumerate(self._stash):
            if stash_key == key:
                self._stash[i] = (key, new_value)
                return None

        raise KeyError(f"Key {key!r} not found in cuckoo table")

    @property
    def stash(self) -> list[tuple[bytes, bytes]]:
        """Get stash entries (items that failed normal insertion)."""
        return list(self._stash)

    def to_database(self) -> list[bytes]:
        """
        Convert to dense database for PIR.

        Returns:
            List of n entries, each of size entry_size
        """
        empty = bytes(self.params.entry_size)
        return [
            bucket[0] + bucket[1] if bucket is not None else empty
            for bucket in self._buckets
        ]

    @classmethod
    def build(
        cls, kv_pairs: dict[bytes, bytes], params: CuckooParams, seed: Optional[bytes] = None
    ) -> "CuckooTable":
        """
        Build cuckoo table from key-value pairs.

        Items that fail normal insertion go to the stash.

        Args:
            kv_pairs: Dictionary mapping unique keys to values
            params: Cuckoo parameters
            seed: Optional seed for hash functions

        Returns:
            CuckooTable with all pairs inserted (some may be in stash)
        """
        table = cls(params, seed)

        for key, value in kv_pairs.items():
            table.insert(key, value)

        return table
