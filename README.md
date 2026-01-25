# RMS24: A Runnable Specification

A readable, executable specification of the RMS24 single-server PIR scheme with Keyword PIR support.

## Purpose

This repository serves as:

- **Runnable Spec**: A readable, executable specification prioritizing clarity over performance
- **Anchor for AI-assisted development**: A reference implementation for LLM-assisted porting to other languages or optimization

> **Note**: `demo.py` and `tests/` are AI-generated and have not been reviewed by humans. The core implementation in `pir/` has been reviewed.

## Features

- **Single-server PIR** with client-dependent preprocessing
  - Based on [RMS24](https://eprint.iacr.org/2023/1072)
  - O(√num_entries) online communication and server computation
  - Supports O(√num_entries) queries before re-running offline phase

- **Keyword PIR** via cuckoo hashing
  - Based on [ALPRSSY21](https://eprint.iacr.org/2019/1483)

- **Updatability** without re-running offline phase

- **Batch operations** for queries and updates

- **Configurable tradeoffs** via `block_size` and `num_backup_hints`

- **Vitalik's optimization** for reduced query communication
  - Original: send offsets for both subsets (c offsets total)
  - Optimized: share offsets between subsets (c/2 offsets total)
  - 50% reduction in query size

## Quick Start

```bash
# Run demo (1 MiB database)
python3 demo.py

# Run Keyword PIR demo
python3 demo.py --kpir
```

## Usage

### Index PIR

```python
from pir.rms24 import Params, Client, Server

database = [bytes(32) for _ in range(1000)]  # 1000 entries, 32 bytes each

params = Params(num_entries=len(database), entry_size=32)
server = Server(database, params)
client = Client(params)

client.generate_hints(server.stream_database())  # Offline

queries = client.query([42, 100, 999])           # Online (batch)
responses = server.answer(queries)
results = client.extract(responses)
client.replenish_hints()

updates = server.update_entries({0: bytes(32), 5: bytes(32)})  # Update (batch)
client.update_hints(updates)
```

### Keyword PIR

```python
from pir.rms24 import Client, Server
from pir.keyword_pir import KPIRParams, KPIRClient, KPIRServer

kv_store = {b"key1".ljust(32): b"value1".ljust(32)}  # 32-byte keys and values

kw_params = KPIRParams(num_items_expected=len(kv_store), key_size=32, value_size=32)
server = KPIRServer.create(kv_store, kw_params, Server)
client = KPIRClient.create(kw_params, Client)

client.generate_hints(server.stream_database())  # Offline

key1, key2 = b"key-0".ljust(32), b"key-1".ljust(32)
queries = client.query([key1, key2])             # Online (batch)
responses, stash = server.answer(queries)
results = client.extract(responses, stash)
client.replenish_hints()

updates = server.update({b"key1".ljust(32): b"new_val".ljust(32)})  # Update
client.update_hints(updates)
```

## Performance

**Setup**: 1 MiB database, 32-byte entries, security_param=80, Apple M3

*Note: This is un-optimized Python prioritizing clarity over performance.*

### Index PIR

| Phase | Operation | Time | Communication |
|-------|-----------|------|---------------|
| Offline | `generate_hints()` | ~23s | 1 MiB (download) |
| Online | `query()` | ~1.1ms | 387 B (mask + offsets) |
| | `answer()` | ~0.26ms | 64 B (2 parities) |
| | `extract()` | ~3μs | - |
| | `replenish_hints()` | ~5μs | - |
| | **Total** | **~1.4ms** | **~450 B** |
| Update | `update_entries()` | ~10μs | 36 B (index + delta) |
| | `update_hints()` | ~43ms | - |

### Keyword PIR

| Phase | Operation | Time | Communication |
|-------|-----------|------|---------------|
| Offline | `generate_hints()` | ~80s | 6 MiB (download) |
| Online | `query()` | ~3.6ms | 1.3 KiB (2 PIR queries) |
| | `answer()` | ~1.4ms | 256 B (2 PIR responses) |
| | `extract()` | ~8μs | - |
| | `replenish_hints()` | ~11μs | - |
| | **Total** | **~5.0ms** | **~1.6 KiB** |
| Update | `update()` | ~28μs | 68 B (index + delta) |
| | `update_hints()` | ~73ms | - |

## Project Structure

```
rms24/
├── demo.py                 # Benchmarks and usage examples
├── pir/
│   ├── protocols.py        # PIR client/server interfaces
│   ├── rms24/              # Core RMS24 implementation
│   │   ├── params.py       # Parameter configuration
│   │   ├── client.py       # Client (hint generation, queries)
│   │   ├── server.py       # Server (database, responses)
│   │   ├── messages.py     # Query/Response types
│   │   └── utils.py        # PRF and XOR utilities
│   └── keyword_pir/        # Keyword PIR layer
│       ├── params.py       # KPIR parameters
│       ├── client.py       # KPIR client
│       ├── server.py       # KPIR server
│       └── cuckoo.py       # Cuckoo hashing
└── tests/                  # Test suite
```

## Optimization Opportunities

The Python implementation prioritizes clarity (e.g., HMAC-SHA256 for PRF). For production use, consider:

- **Faster PRF**: Use ChaCha or AES instead of HMAC-SHA256
- **Batched PRF evaluation**: Compute multiple PRF outputs per call
- **Partial sort for cutoff**: Use `nth_element` / quickselect instead of full sort (O(n) vs O(n log n))
- **SIMD XOR**: Use AVX2/AVX-512 for parity accumulation
- **Parallel hint generation**: Process multiple hints concurrently
- **KPIR expansion factor**: Current setting is conservative; can be tuned for specific use cases
- **KPIR entry size**: Store `Hash(Key)||Value` instead of `Key||Value` to reduce entry size

## References

- **RMS24**: Ling Ren, Muhammad Haris Mughees, I Sun. [Simple and Practical Amortized Sublinear Private Information Retrieval](https://eprint.iacr.org/2023/1072). CCS 2024.
- **S3PIR**: [github.com/renling/S3PIR](https://github.com/renling/S3PIR) - Accompanying C++ PoC for RMS24 with optimizations (batched AES-PRF).
- **Keyword PIR**: Asra Ali, Tancrède Lepoint, Sarvar Patel, Mariana Raykova, Phillipp Schoppmann, Karn Seth, Kevin Yeo. [Communication-Computation Trade-offs in PIR](https://eprint.iacr.org/2019/1483). USENIX Security 2021.

## License

Apache 2.0
