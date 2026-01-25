# Keyword PIR (KPIR)

Keyword PIR layer that enables key-value lookups on top of any index PIR scheme, based on [Section 5 of eprint 2019/1483](https://eprint.iacr.org/2019/1483).

## Overview

Keyword PIR converts keyword queries to index queries using cuckoo hashing:

1. Server builds a cuckoo hash table from key-value pairs
2. Client queries multiple bucket positions for each keyword
3. Client checks which bucket contains the matching key

## Parameters

```python
KPIRParams(
    num_items_expected=n,     # Expected number of key-value pairs
    key_size=32,              # Key size in bytes
    value_size=32,            # Value size in bytes
    num_hashes=2,             # Cuckoo hash functions (default: 2)
    expansion_factor=3,       # Buckets = 3 × items (default)
    max_evictions=100,        # Before using stash
)
```

**Derived values:**
- `num_buckets` = expansion_factor × num_items_expected
- `entry_size` = key_size + value_size

## Files

| File | Description |
|------|-------------|
| `params.py` | KPIR parameters wrapping cuckoo configuration |
| `client.py` | Keyword-to-index translation, result extraction |
| `server.py` | Cuckoo table management, query forwarding |
| `cuckoo.py` | Cuckoo hashing implementation |

## Cuckoo Hashing

Each key-value pair is stored in one of `num_hashes` possible bucket positions:

```
bucket_positions(key) = [H_0(key) % num_buckets, H_1(key) % num_buckets, ...]
```

**Insertion**: Try each position; if all occupied, evict an existing item and re-insert it (up to `max_evictions`). Items that can't be placed go to a stash.

**Lookup**: Query all `num_hashes` positions and check which contains the key.

## Protocol

### Setup

```
Server:
  1. Build cuckoo table from key-value pairs
  2. Convert table to PIR database (bucket → entry)
  3. Initialize underlying PIR server

Client:
  1. Initialize cuckoo hasher with same seed
  2. Initialize underlying PIR client
```

### Query

```
Client                                  Server
  |                                       |
  [compute bucket positions for keyword]  |
  [generate num_hashes PIR queries]       |
  |                                       |
  |  ---- num_hashes PIR queries -------> |
  |                                       |
  |        [answer each PIR query]        |
  |                                       |
  |  <--- num_hashes responses + stash -- |
  |                                       |
  [extract entries from responses]        |
  [find entry where key matches keyword]  |
  [return value or None if not found]     |
```

## Communication Costs

The underlying PIR operates on `expansion_factor × n` buckets (default: 3n).

Per keyword query:
- **Query**: num_hashes × PIR query size
- **Response**: num_hashes × PIR response size + stash

With 2 hash functions, expansion_factor=3, and RMS24:
- Query: 2 × O(√3n) bytes
- Response: 2 × O(entry_size) bytes

## Example

```python
from pir.rms24 import Client, Server
from pir.keyword_pir import KPIRParams, KPIRClient, KPIRServer

# Key-value store (32-byte keys and values)
kv_store = {
    b"alice".ljust(32): b"alice_value".ljust(32),
    b"bob".ljust(32): b"bob_value".ljust(32),
}

# Parameters
kw_params = KPIRParams(
    num_items_expected=len(kv_store),
    key_size=32,
    value_size=32,
)

# Setup
server = KPIRServer.create(kv_store, kw_params, Server)
client = KPIRClient.create(kw_params, Client)

# Offline phase
client.generate_hints(server.stream_database())

# Query by keyword
queries = client.query([b"alice".ljust(32), b"bob".ljust(32)])
responses, stash = server.answer(queries)
values = client.extract(responses, stash)  # Returns value or None for each
client.replenish_hints()
```

## Updates

Supports insert, update, and delete operations:

```python
# Update existing key
updates = server.update({b"alice".ljust(32): b"new_value".ljust(32)})
client.update_hints(updates)

# Insert new key
updates = server.update({b"charlie".ljust(32): b"charlie_value".ljust(32)})
client.update_hints(updates)

# Delete key
updates = server.update({b"alice".ljust(32): None})
client.update_hints(updates)
```

## Stash

Items that can't be placed in the cuckoo table go to a small stash. The stash is sent with every response. With default parameters (expansion_factor=3, num_hashes=2), the stash is typically empty for reasonable load factors.

## Capacity

Each keyword query consumes `num_hashes` PIR queries internally:

```python
remaining_keyword_queries = client.remaining_queries()
# = underlying_pir.remaining_queries() // num_hashes
```
