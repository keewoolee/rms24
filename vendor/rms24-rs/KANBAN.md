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

- [x] Benchmark Forge kernel on Modal H200 (2026-01-25)
  - Previous (old kernel): 98 hints/sec
  - Forge kernel (256 threads): **985K hints/sec per GPU**
  - Production run (λ=128): **10.97M hints** in **5.4 min** on 50× H200
  - Combined throughput: **49.2M hints/sec**

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

**Subset generation optimizations:**
- [x] Vectorized PyTorch generation (2026-01-25)
  - Generate all indices at once instead of Python loop
  - Reduced subset gen from 90s to 65s (28% faster)
  - Wall clock: 5.4min → 4.6min, Cost: $19 → $16
- [ ] Rust + rayon parallel generation (est. 5-10s)
- [x] CUDA kernel for PRF-based subset gen (2026-01-25)
  - `hint_gen_with_prf_kernel`: ChaCha12 PRF on GPU
  - Eliminates 65s CPU subset gen + 37.6GB transfer per GPU
  - Script: `scripts/modal_forge_gpu_subset.py`
  - Kernel: `cuda/subset_gen_kernel.cu`

**GPU kernel optimizations:**
- [x] Precompute subset lists on CPU, pass to GPU
  - Implemented: `HintSubset` + `SubsetData` structs
  - GPU receives flattened block/offset arrays
- [x] Warp-level parallelism (`rms24_hint_gen_warp_kernel`)
  - 32 threads per hint, strided block processing
  - Butterfly shuffle reduction (`__shfl_xor_sync`)
- [x] Forge-optimized kernel (256 threads, 8 warps)
  - Two-level XOR reduction (warp shuffle + shared memory)
  - 985K hints/sec per GPU (370× faster than Plinko)
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
| 10x H200, shared Phase 1 | 1,000 | 5.17s | 193 hints/sec |
| 10x H200, shared Phase 1 | 10,000 | 4.90s | 2,042 hints/sec |
| 20x H200, shared Phase 1 | 20,000 | 4.34s | 4,610 hints/sec |
| 50x H200, shared Phase 1 | 50,000 | 2.19s | **22,855 hints/sec**

**Forge-optimized kernel (256 threads, 2026-01-25):**
| Config | Hints | Kernel Time | Throughput |
|--------|-------|-------------|------------|
| H200, 10M entries, S=512 | 100 | 0.024 ms | 4.1M hints/sec |
| H200, 10M entries, S=1024 | 1,000 | 0.053 ms | 18.9M hints/sec |
| 50x H200, 10M entries, S=1024 | 100,000 | 0.094 ms avg | **1.06B hints/sec combined** |

**Production run (λ=128, 2026-01-25):**
| Config | Hints | Kernel Time | Throughput |
|--------|-------|-------------|------------|
| 50x H200, 73GB mainnet, S=21K | 10,966,016 | 222.7 ms avg | **49.2M hints/sec combined** |
| Per-GPU (best) | 219,320 | 221.7 ms | 989K hints/sec |
| Per-GPU (worst) | 219,320 | 231.4 ms | 948K hints/sec |
| Wall clock time | - | - | **5.4 min** |

**Production run with vectorized subset gen (λ=128, 2026-01-25):**
| Config | Hints | Kernel Time | Throughput |
|--------|-------|-------------|------------|
| 50x H200, 73GB mainnet, S=21K | 10,966,016 | 222.4 ms avg | **49.3M hints/sec combined** |
| Subset gen time | - | 65s avg (was 90s) | **28% faster** |
| Wall clock time | - | - | **4.6 min (was 5.4 min)** |
| Cost | - | - | **~$16 (was ~$19)** |

**Production run with GPU PRF (λ=128, 2026-01-25):**
| Config | Hints | Kernel Time | Throughput |
|--------|-------|-------------|------------|
| 50x H200, 73GB mainnet, GPU PRF | 10,966,016 | 411.2 ms avg | **26.7M hints/sec combined** |
| Subset gen time | - | **0s** | **Eliminated!** |
| Data transfer | - | **0 GB** | **37.6GB/GPU eliminated** |
| Wall clock time | - | - | **4.5 min** |
| Per-GPU throughput | 219,320 | 411ms | **533K hints/sec** |

**Distributed (shared Phase 1):**
- Phase 1 (CPU, 32 cores): 2.4s for 26K hints on 100MB DB
- Phase 2 scales linearly with GPU count
- Individual GPU kernel: 0.3-0.5s per 1K hints (after cargo cache)

**Bottlenecks remaining:**
1. ~~Phase 1 CPU still slow~~ - Vectorized generation now 65s (was 90s)
2. GPU throughput scales with batch size (more hints = better occupancy)
3. Memory transfer overhead: 37.6 GB subset data per GPU

**Cost breakdown (λ=128 production run):**
| Phase | Time (CPU subset) | Time (GPU PRF) | Reusable? |
|-------|-------------------|----------------|-----------|
| DB load | 85s | 85s | Yes (keep GPU warm) |
| Kernel compile | 42s | 42s | Yes (pre-compile) |
| Subset generation | 65s | **0s** | No (per-client PRF key) |
| Subset data transfer | ~10s | **0s** | No (37.6GB per GPU) |
| Kernel execution | 0.22s | **0.41s** | No (per-client) |
| **Total per client** | **75.2s** | **0.41s** | |
| **Total cold start** | **192s** | **~128s** | |

**Modal H200 pricing:**
- GPU: $0.001261/sec, CPU: $0.0000131/core/sec (17 cores)
- Cold start: ~$16 for 50 GPUs
- Warm (persistent): ~$4.10 per client (subset gen + kernel only)
