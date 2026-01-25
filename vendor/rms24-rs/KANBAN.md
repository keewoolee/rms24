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

- [ ] Benchmark warp kernel on Modal H200 (2026-01-25)
  - Previous (old kernel): 98 hints/sec
  - New: Precomputed subsets + warp parallelism implemented
  - Run: `modal run scripts/modal_run_bench.py --gpu h200 --max-hints 1000`

## Next Up

### GPU Validation
- [ ] Verify CPU/GPU parity consistency (same key -> same parities)
- [ ] Benchmark warp kernel vs old kernel throughput
- [ ] Test with mainnet-v3 dataset (73GB, 1.8B entries)
- [ ] Implement backup hint high parity in warp kernel (currently TODO)

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

### Optimizations (Priority Order)

**Phase 1 CPU optimizations:**
- [ ] Parallelize Phase 1 with rayon (currently single-threaded)
- [ ] Cache select_vector results (reused across hints)

**GPU kernel optimizations:**
- [x] Precompute subset lists on CPU, pass to GPU
  - Implemented: `HintSubset` + `SubsetData` structs
  - GPU receives flattened block/offset arrays
- [x] Warp-level parallelism (`rms24_hint_gen_warp_kernel`)
  - 32 threads per hint, strided block processing
  - Butterfly shuffle reduction (`__shfl_xor_sync`)
- [ ] Vectorized loads (ulong2) for 16-byte aligned reads
- [ ] Coalesced memory access patterns (sort blocks by stride)

**Infrastructure:**
- [ ] Multi-GPU support in Modal script
- [ ] Streaming hint generation (don't load full 73GB into GPU memory)

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

**Old kernel (PRF on GPU):**
| Config | Hints | GPU Time | Throughput |
|--------|-------|----------|------------|
| H200, 73GB mainnet, 42K blocks | 1,000 | 10.2s | 98 hints/sec |

**Warp kernel (precomputed subsets):**
| Config | Hints | GPU Time | Throughput |
|--------|-------|----------|------------|
| H200, 100MB synthetic, 1.6K blocks | 100 | 51ms | 1,974 hints/sec |
| H200, 1GB synthetic, 5.1K blocks | 1,000 | 1.17s | 854 hints/sec |
| 10x H200, 100MB synthetic (parallel) | 1,000 | 1.78s | 563 hints/sec |

**Bottlenecks remaining:**
1. Phase 1 CPU still slow (66s for 1K hints on 5K blocks) - rayon helps but PRF is sequential
2. GPU throughput scales with batch size (more hints = better occupancy)
3. Memory transfer overhead for small batches
