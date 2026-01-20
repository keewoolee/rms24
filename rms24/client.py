import secrets
from dataclasses import dataclass
from typing import Iterator, Optional

from .params import Params
from .protocol import Query, Response, EntryUpdate
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
        self._query_states: list[_QueryState] = []

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
        self._query_states = []

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

    def query(self, indices: list[int]) -> list[Query]:
        """
        Prepare queries for multiple indices.

        Note: Failure probability increases with batch size.
        Each query requires a distinct hint containing its index.

        Args:
            indices: List of database indices to retrieve

        Returns:
            List of queries to send to server
        """
        if not self.hints.reg_hints:
            raise RuntimeError("Must call generate_hints() before querying")

        if self._query_states:
            raise RuntimeError("Previous query batch not yet completed")

        c = self.params.c
        w = self.params.w
        n = len(indices)

        # Precompute (block, offset) for all indices
        targets = [
            (self.params.block_of(idx), self.params.offset_in_block(idx))
            for idx in indices
        ]

        # Results indexed by position (to preserve order)
        queries: list[Query | None] = [None] * n
        states: list[_QueryState | None] = [None] * n
        remaining: set[int] = set(range(n))  # positions not yet matched

        # Flipped loop: iterate hints, check remaining indices
        for hint_pos, hint in enumerate(self.hints.reg_hints):
            if not remaining:
                break

            # Check if this hint matches any remaining index
            matched_pos = None
            for i in remaining:
                if hint.contains(targets[i][0], targets[i][1], self.prf, w):
                    matched_pos = i
                    break

            if matched_pos is None:
                continue

            remaining.remove(matched_pos)
            queried_block, queried_offset = targets[matched_pos]
            j = hint.hint_id
            extra_block, extra_offset = hint.extra

            # Build mask and offsets in a single pass (Vitalik Buterin's trick)
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

            queries[matched_pos] = Query(mask=mask, offsets=offsets)
            states[matched_pos] = _QueryState(
                real_is_first=real_is_first,
                hint=hint,
                hint_pos=hint_pos,
                queried=(queried_block, queried_offset),
            )

        if remaining:
            unmatched = [indices[i] for i in remaining]
            raise RuntimeError(f"No hints found for indices: {unmatched}")

        self._query_states = states

        return queries

    def extract(self, responses: list[Response]) -> list[bytes]:
        """
        Extract results from server responses.

        Args:
            responses: List of responses from server

        Returns:
            List of database entries at the queried indices
        """
        if not self._query_states:
            raise RuntimeError("Must call query() before extract()")

        if len(responses) != len(self._query_states):
            raise RuntimeError(
                f"Response count ({len(responses)}) doesn't match query count ({len(self._query_states)})"
            )

        results = []
        for state, response in zip(self._query_states, responses):
            real_parity = response.parity_0 if state.real_is_first else response.parity_1
            entry = xor_bytes(real_parity, state.hint.parity)
            state.entry = entry
            results.append(entry)

        return results

    def replenish_hints(self) -> None:
        """
        Replenish all consumed hints using backup hints.

        Must be called after extract() to complete the query batch.
        """
        if not self._query_states:
            raise RuntimeError("Must call query() before replenish_hints()")

        for state in self._query_states:
            if state.entry is None:
                raise RuntimeError("Must call extract() before replenish_hints()")

        if len(self.hints.backup_hints) < len(self._query_states):
            raise RuntimeError(
                f"Not enough backup hints ({len(self.hints.backup_hints)}) "
                f"for {len(self._query_states)} queries"
            )

        for state in self._query_states:
            backup = self.hints.backup_hints.pop()

            queried_block, queried_offset = state.queried
            queried_entry = state.entry

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
            self.hints.reg_hints[state.hint_pos] = new_hint

        self._query_states = []

    def update_hints(self, updates: list[EntryUpdate]) -> None:
        """
        Update hints affected by database entry changes.

        Args:
            updates: List of EntryUpdate containing index and delta
        """
        if not self.hints.reg_hints:
            raise RuntimeError("Must call generate_hints() before update_hints()")

        if not updates:
            return

        w = self.params.w

        # Group updates by block: block -> {offset: delta}
        by_block: dict[int, dict[int, bytes]] = {}
        for u in updates:
            block = self.params.block_of(u.index)
            offset = self.params.offset_in_block(u.index)
            if block not in by_block:
                by_block[block] = {}
            by_block[block][offset] = u.delta

        # Update regular hints
        for hint in self.hints.reg_hints:
            ex_block, ex_offset = hint.extra

            # Check extra
            if ex_block in by_block and ex_offset in by_block[ex_block]:
                hint.parity = xor_bytes(hint.parity, by_block[ex_block][ex_offset])

            # Check selected blocks (one PRF call per block)
            for block, offsets in by_block.items():
                v = self.prf.select(hint.hint_id, block)
                if not block_selected(v, hint.cutoff, hint.flip):
                    continue
                r = self.prf.offset(hint.hint_id, block) % w
                if r in offsets:
                    hint.parity = xor_bytes(hint.parity, offsets[r])

        # Update backup hints
        for backup in self.hints.backup_hints:
            for block, offsets in by_block.items():
                v = self.prf.select(backup.hint_id, block)
                r = self.prf.offset(backup.hint_id, block) % w

                if r in offsets:
                    if v < backup.cutoff:
                        backup.parity_low = xor_bytes(backup.parity_low, offsets[r])
                    else:
                        backup.parity_high = xor_bytes(backup.parity_high, offsets[r])

    def remaining_queries(self) -> int:
        """Return number of queries remaining before offline phase needed."""
        return len(self.hints.backup_hints)
