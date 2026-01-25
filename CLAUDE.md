# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A readable, executable specification of the RMS24 single-server PIR (Private Information Retrieval) scheme with Keyword PIR support. This is intentionally un-optimized Python that prioritizes clarity over performance.

**Important**: `demo.py` and `tests/` are AI-generated and unreviewed. The core implementation in `pir/` has been human-reviewed.

## Commands

```bash
# Run demo (1 MiB database)
python3 demo.py

# Run Keyword PIR demo
python3 demo.py --kpir

# Run both demos
python3 demo.py --all

# Run with custom parameters
python3 demo.py --entries 1000000 --entry-size 64 --queries 20

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_correctness.py

# Run a specific test
pytest tests/test_correctness.py::TestParams::test_basic_params
```

## Architecture

### Protocol Flow (Client-Dependent Preprocessing Model)

1. **Offline Phase**: Server streams database → Client generates hints
2. **Online Phase**: `client.query()` → `server.answer()` → `client.extract()` → `client.replenish_hints()`
3. **Updates**: `server.update_entries()` → `client.update_hints()`

### Module Structure

- `pir/protocols.py` - Abstract interfaces (`PIRClient`, `PIRServer`, `PIRParams`) that define the Piano-like PIR pattern
- `pir/rms24/` - Core RMS24 implementation
  - `params.py` - Parameter computation (block_size, num_blocks, hints)
  - `client.py` - Hint generation and query logic
  - `server.py` - Database and query answering
  - `messages.py` - Query/Response dataclasses
  - `utils.py` - HMAC-based PRF and XOR utilities
- `pir/keyword_pir/` - Keyword PIR layer using cuckoo hashing
  - Wraps any index PIR to support key-value lookups
  - Each keyword query generates `num_hashes` (default: 2) underlying PIR queries

### Key Data Structures

**HintState** (in client.py): Stores hints using parallel arrays indexed by hint_id:
- `cutoffs[i]`: Median value splitting blocks (0 = consumed)
- `parities[i]`: XOR of entries in hint's subset
- `extra_blocks[i]`, `extra_offsets[i]`: Extra entry from unselected block
- `flips[i]`: Selection direction (inverted after backup promotion)
- `backup_parities_high[i]`: Second parity for backup hints

**Params** key relationships:
- `block_size` defaults to `√num_entries`
- `num_blocks` = `⌈num_entries / block_size⌉`, rounded to even
- `num_reg_hints` = `security_param × block_size`
- Supports O(√num_entries) queries per offline phase

### Vitalik's Optimization

Query offsets are shared between the two subsets (c/2 offsets instead of c), reducing query communication by 50%.
