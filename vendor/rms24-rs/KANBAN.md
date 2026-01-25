# RMS24-RS Kanban

## Done

- [x] Project scaffolding (Cargo.toml, lib.rs, params.rs)
- [x] ChaCha12-based PRF module
- [x] Hint state data structures
- [x] CPU hint generation (Phase 1 + Phase 2)
- [x] CUDA kernel with ChaCha12
- [x] GPU module (Rust bindings via cudarc)
- [x] Integration tests (17 passing)
- [x] Benchmarks (CPU hint gen, PRF ops)
- [x] Modal GPU benchmark script

## In Progress

- [x] Test GPU kernel on Modal H200 (2026-01-25)
  - 1K hints on 73GB mainnet: 10.2s avg (98 hints/sec)
  - Bottleneck: 42K blocks per hint, no warp optimization
  - Next: Add precomputed subsets + warp parallelism

## Next Up

### GPU Validation
- [ ] Verify CPU/GPU parity consistency (same key -> same parities)
- [ ] Benchmark GPU vs CPU throughput
- [ ] Test with mainnet-v3 dataset (73GB, 1.8B entries)

### Protocol Completion
- [ ] Server module (query answering)
- [ ] Query/Response messages
- [ ] Client query generation
- [ ] Client extract + replenish hints
- [ ] Hint updates (delta application)

### Keyword PIR Layer
- [ ] Port cuckoo hashing from Python
- [ ] Keyword client/server wrappers
- [ ] 8-byte TAG fingerprint verification

### Optimizations
- [ ] Warp-level parallelism in CUDA kernel (like Plinko)
- [ ] Vectorized loads (ulong2) for 16-byte aligned reads
- [ ] Multi-GPU support in Modal script
- [ ] Pre-derive PRP keys on CPU to reduce GPU work

### Testing
- [ ] Property tests (proptest) for PRF
- [ ] Fuzz testing for hint generation
- [ ] Cross-validation with Python reference implementation

### Documentation
- [ ] API docs (cargo doc)
- [ ] Protocol description in README
- [ ] Benchmark results table

## Blocked

- [ ] Production hint generation (needs GPU validation first)

## Notes

### Modal Commands

```bash
# Quick test (100K hints)
modal run scripts/modal_run_bench.py --gpu h200 --max-hints 100000

# Full benchmark
modal run scripts/modal_run_bench.py --gpu h200 --lambda 80 --iterations 5

# With mainnet data
modal run scripts/modal_run_bench.py --gpu h200 --db /data/mainnet-v3/database.bin
```

### Key Differences from Plinko

| Aspect | Plinko | RMS24-RS |
|--------|--------|----------|
| Subset selection | iPRF (SwapOrNot PRP) | Median cutoff |
| PRF | ChaCha + SwapOrNot | ChaCha12 only |
| Hint structure | parity only | cutoff + parity + extra |
| Rounds | 759 SwapOrNot | 0 (direct PRF) |

### Performance Targets

- CPU PRF: 139 ns/call (achieved)
- GPU hint gen: 98 hints/sec (current, unoptimized)
- Target: Match Plinko's ~500K hints/sec on H200

### Benchmark Results (2026-01-25)

| Config | Hints | GPU Time | Throughput |
|--------|-------|----------|------------|
| H200, 73GB mainnet, 42K blocks | 1,000 | 10.2s | 98 hints/sec |

**Bottlenecks identified:**
1. No precomputed subsets (each hint checks all 42K blocks)
2. No warp-level parallelism (Plinko shares key loads)
3. Phase 1 CPU slow (4.3s for 1K hints)
