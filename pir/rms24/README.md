# RMS24 Single-Server PIR

Implementation of the RMS24 scheme from ["Simple and Practical Amortized Sublinear Private Information Retrieval"](https://eprint.iacr.org/2023/1072).

## Overview

RMS24 achieves sublinear online query time through client-dependent preprocessing:

- **Offline**: Client streams database to build hints
- **Online**: Each query uses O(√num_entries) communication and server computation
- **Capacity**: Supports O(√num_entries) queries before re-running offline phase

## Parameters

```python
Params(
    num_entries=1000,        # Database size
    entry_size=32,           # Bytes per entry
    security_param=80,       # Failure probability ≈ 2^{-security_param}
    block_size=None,         # Default: ⌈√num_entries⌉
    num_backup_hints=None,   # Default: security_param × block_size
)
```

**Derived values:**
- `num_blocks` = ⌈num_entries / block_size⌉, rounded to even
- `num_reg_hints` = security_param × block_size
- `num_backup_hints` = number of queries supported per offline phase

**Tradeoffs:**
- Larger `block_size` → smaller queries, more hints to store
- Larger `security_param` → lower failure probability, more hints
- Larger `num_backup_hints` → more queries per offline phase, more storage
- Default `block_size = √num_entries` balances query size and storage to O(√num_entries)

## Files

| File | Description |
|------|-------------|
| `params.py` | Parameter configuration and validation |
| `client.py` | Hint generation, query preparation, result extraction |
| `server.py` | Database storage, query answering |
| `messages.py` | Query and Response dataclasses |
| `utils.py` | HMAC-based PRF and XOR utilities |

## Hint Structure

Hints are stored in `HintState` using parallel arrays (index = hint_id):

**Regular hints** (indices 0 to num_reg_hints-1):
- `cutoffs[i]`: Median value splitting blocks into two halves (0 = consumed)
- `extra_blocks[i]`, `extra_offsets[i]`: Extra entry from an unselected block
- `parities[i]`: XOR of entries in the hint's subset
- `flips[i]`: Selection direction (inverted after promotion from backup)

A regular hint's subset contains `num_blocks/2 + 1` entries:
- One entry from each "selected" block (where `PRF(hint_id, block) < cutoff`, or `>= cutoff` if flipped)
- One extra entry from an "unselected" block

**Backup hints** (indices num_reg_hints to num_total_hints-1):
- `cutoffs[i]`: Same as regular hints
- `parities[i]`: XOR of entries from blocks where `PRF < cutoff`
- `backup_parities_high[i - num_reg_hints]`: XOR of entries from blocks where `PRF >= cutoff`

When a backup hint is promoted to regular, it picks one of the two parities based on whether the queried block is selected, and sets `flips[i]` accordingly.

## Protocol

### Offline Phase

```
Client                                  Server
  |                                       |
  |  <------- blocks[0..num_blocks-1] --- |
  |                                       |
  [generate_hints: build num_reg_hints    |
   regular + backup hints]                |
```

### Online Phase (Query)

```
Client                                  Server
  |                                       |
  [find hint containing index i]          |
  [build query hiding i in hint subset]   |
  |                                       |
  |  -------- Query(mask, offsets) -----> |
  |    mask: bitmask assigning blocks     |
  |          to subset 0 or 1             |
  |    offsets: position within each block|
  |                                       |
  |           [compute parity_0, parity_1:|
  |            XOR of entries at offsets  |
  |            for each subset]           |
  |                                       |
  |  <------- Response(p0, p1) ---------- |
  |                                       |
  [extract: result = parity XOR hint.parity]
  [replenish: convert backup → regular]   |
```

## Security

The query hides which entry is accessed:
1. Both subsets in Query have exactly `num_blocks/2` blocks
2. The mask randomly assigns the real subset to position 0 or 1
3. Offsets are shared between subsets, revealing no information
4. Server sees two equal-sized subsets and cannot distinguish which is real

The PRF key is client-secret. All hint data must remain private.

## Communication Costs

| Message | Size |
|---------|------|
| Query | ⌈num_blocks/8⌉ + num_blocks bytes (mask + offsets) |
| Response | 2 × entry_size bytes |

With default `block_size = √num_entries`:
- Query: O(√num_entries) bytes
- Response: O(entry_size) bytes

## Example

```python
from pir.rms24 import Params, Client, Server

# 4MB database: 131072 entries × 32 bytes
params = Params(num_entries=131072, entry_size=32)
# block_size=362, num_blocks=364, num_reg_hints=28960

database = [bytes(32) for _ in range(131072)]
server = Server(database, params)
client = Client(params)

client.generate_hints(server.stream_database())  # Offline

queries = client.query([42, 100, 999])           # Online
responses = server.answer(queries)
results = client.extract(responses)
client.replenish_hints()

print(f"Remaining queries: {client.remaining_queries()}")
```

## Updates

Database updates are supported without re-running offline phase:

```python
# Server updates entries
updates = server.update_entries({0: new_value, 5: another_value})

# Client updates affected hints
client.update_hints(updates)
```

The `delta` in each update is `old_value XOR new_value`, allowing efficient hint adjustment.
