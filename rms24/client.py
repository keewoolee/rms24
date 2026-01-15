import secrets
from dataclasses import dataclass
from typing import Iterator, Optional

from .params import Params
from .protocol import Query, Response
from .prf import PRF, find_median_cutoff, block_selected
from .hint import RegHint, BackupHint, HintStorage
from .utils import xor_bytes, zero_entry


@dataclass
class _QueryState:
    """Internal state from query() needed for extract() and replenish()."""
    queried: tuple[int, int]  # (block, offset)
    hint: RegHint
    hint_pos: int
    real_is_first: bool
    entry: Optional[bytes] = None  # Set by extract()


class Client:
    """
    PIR Client for the single-server RMS24 scheme.

    The client maintains hints that allow sublinear online queries.
    After num_backup_hints queries, the offline phase must be re-run.
    """

    def __init__(self, params: Params):
        """
        Initialize client.

        Args:
            params: PIR parameters
        """
        self.params = params
        self.prf: Optional[PRF] = None
        self.hints = HintStorage()
        self._query_state: Optional[_QueryState] = None

    def generate_hints(self, db_stream: Iterator[tuple[int, list[bytes]]]) -> None:
        """
        Processes the database block by block and constructs:
        - num_reg_hints regular hints
        - num_backup_hints backup hints

        Args:
            db_stream: Iterator yielding (block_id, entries) tuples
        """
        self.prf = PRF()
        self.hints = HintStorage()
        self._query_state = None

        num_reg_hints = self.params.num_reg_hints
        c = self.params.c  # number of blocks
        w = self.params.w  # entries per block

        num_total_hints = num_reg_hints + self.params.num_backup_hints

        # For each hint, compute V_j and find median cutoff
        # Retry with different hint_id if collision at median (cutoff is None)
        hint_ids = []  # actual hint_id used for each hint
        cutoffs = []
        extras = []  # (block, offset) pairs for regular hints only
        extras_by_block = [[] for _ in range(c)]

        j = 0
        while len(hint_ids) < num_total_hints:
            V_j = self.prf.select_vector(j, c)
            cutoff, unselected = find_median_cutoff(V_j)

            if cutoff is None:
                # Collision at median, skip this hint_id
                j += 1
                continue

            hint_idx = len(hint_ids)
            hint_ids.append(j)
            cutoffs.append(cutoff)

            # For regular hints, pick an extra (block, offset) from an unselected block
            if hint_idx < num_reg_hints:
                extra_block = secrets.choice(unselected)
                extra_offset = secrets.randbelow(w)
                extras.append((extra_block, extra_offset))
                extras_by_block[extra_block].append((hint_idx, extra_offset))

            j += 1

        # Initialize parities to zero
        entry_size = self.params.entry_size
        parities = [zero_entry(entry_size) for _ in range(num_total_hints)]
        backup_parities_high = [
            zero_entry(entry_size) for _ in range(self.params.num_backup_hints)
        ]

        # Stream database and accumulate parities
        for k, block_entries in db_stream:
            for hint_idx in range(num_total_hints):
                hint_id = hint_ids[hint_idx]
                v_jk = self.prf.select(hint_id, k)
                cutoff = cutoffs[hint_idx]
                r_jk = self.prf.offset(hint_id, k) % w
                x = block_entries[r_jk]

                if v_jk < cutoff:
                    parities[hint_idx] = xor_bytes(parities[hint_idx], x)
                elif hint_idx >= num_reg_hints:
                    backup_idx = hint_idx - num_reg_hints
                    backup_parities_high[backup_idx] = xor_bytes(
                        backup_parities_high[backup_idx], x
                    )

            # XOR extras in this block
            for hint_idx, offset in extras_by_block[k]:
                parities[hint_idx] = xor_bytes(parities[hint_idx], block_entries[offset])

        # Regular hints
        for hint_idx in range(num_reg_hints):
            hint = RegHint(
                hint_id=hint_ids[hint_idx],
                cutoff=cutoffs[hint_idx],
                extra=extras[hint_idx],
                parity=parities[hint_idx],
                flip=False,
            )
            self.hints.reg_hints.append(hint)

        # Backup hints
        for backup_idx in range(self.params.num_backup_hints):
            hint_idx = num_reg_hints + backup_idx
            backup = BackupHint(
                hint_id=hint_ids[hint_idx],
                cutoff=cutoffs[hint_idx],
                parity_low=parities[hint_idx],
                parity_high=backup_parities_high[backup_idx],
            )
            self.hints.backup_hints.append(backup)

    def query(self, index: int) -> Query:
        """
        Prepare online query.

        Args:
            index: Database index to retrieve

        Returns:
            Query to send to server
        """
        if not self.hints.reg_hints:
            raise RuntimeError("Must call generate_hints() before querying")

        if self._query_state is not None:
            raise RuntimeError("Previous query not yet completed")

        queried_block = self.params.block_of(index)
        queried_offset = self.params.offset_in_block(index)
        c = self.params.c
        w = self.params.w

        # Find a hint whose subset contains the queried index
        result = self.hints.find_hint(queried_block, queried_offset, self.prf, w)
        if result is None:
            raise RuntimeError(f"No hint found containing index {index}")

        hint_pos, hint = result
        j = hint.hint_id
        extra_block, extra_offset = hint.extra

        # Build mask and offsets in a single pass (Vitalik Buterin's trick)
        # Offsets are shared by position between real and dummy subsets
        real_is_first = secrets.randbelow(2) == 0
        mask_int = 0
        offsets = []

        for k in range(c):
            if k == queried_block:
                is_real = False
            elif block_selected(self.prf.select(j, k), hint.cutoff, hint.flip):
                is_real = True
            elif k == extra_block:
                is_real = True
            else:
                is_real = False

            if is_real:
                if k == extra_block:
                    offsets.append(extra_offset)
                else:
                    offsets.append(self.prf.offset(j, k) % w)

            if is_real == real_is_first:
                mask_int |= 1 << k

        mask = mask_int.to_bytes((c + 7) // 8, "little")

        self._query_state = _QueryState(
            real_is_first=real_is_first,
            hint=hint,
            hint_pos=hint_pos,
            queried=(queried_block, queried_offset),
        )

        return Query(mask=mask, offsets=offsets)

    def extract(self, response: Response) -> bytes:
        """
        Extract result from server response.

        Args:
            response: Response from server

        Returns:
            Database entry at the queried index
        """
        if self._query_state is None:
            raise RuntimeError("Must call query() before extract()")

        real_parity = response.parity_0 if self._query_state.real_is_first else response.parity_1
        entry = xor_bytes(real_parity, self._query_state.hint.parity)

        # Store entry in state for replenish
        self._query_state.entry = entry

        return entry

    def replenish_hint(self) -> None:
        """
        Replenish the consumed hint using a backup hint.

        Must be called after extract() to complete the query.
        """
        if self._query_state is None:
            raise RuntimeError("Must call query() before replenish_hint()")

        if self._query_state.entry is None:
            raise RuntimeError("Must call extract() before replenish_hint()")

        # Get next unused backup hint
        if not self.hints.backup_hints:
            raise RuntimeError("No backup hints remaining")
        backup = self.hints.backup_hints.pop()

        queried_block, queried_offset = self._query_state.queried
        queried_entry = self._query_state.entry

        # Check which half does NOT contain the queried block
        v = self.prf.select(backup.hint_id, queried_block)

        if v >= backup.cutoff:
            # Queried block is in "high" half (v >= cutoff), use low half
            parity = backup.parity_low
            flip = False
        else:
            # Queried block is in "low" half (v < cutoff), use high half
            parity = backup.parity_high
            flip = True

        # Create new regular hint with queried position as extra
        new_hint = RegHint(
            hint_id=backup.hint_id,
            cutoff=backup.cutoff,
            extra=(queried_block, queried_offset),
            parity=xor_bytes(parity, queried_entry),
            flip=flip,
        )

        # Replace consumed hint
        self.hints.reg_hints[self._query_state.hint_pos] = new_hint
        self._query_state = None

    def remaining_queries(self) -> int:
        """Return number of queries remaining before offline phase needed."""
        return len(self.hints.backup_hints)
