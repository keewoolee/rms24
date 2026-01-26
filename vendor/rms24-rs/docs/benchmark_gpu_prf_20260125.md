# GPU PRF Hint Generation Benchmark

**Date:** 2026-01-25
**Kernel:** `hint_gen_with_prf_kernel` (ChaCha12 PRF on GPU)

## Summary

Benchmark of GPU-based subset generation using ChaCha12 PRF, eliminating CPU subset generation and 37.6GB data transfer per GPU.

**Key Results (Production Run λ=128):**
- **50× H200:** 10,966,016 hints in **4.5 minutes** (wall clock)
- **Total Throughput:** **26.7M hints/sec** (Aggregate)
- **Per-GPU Throughput:** **533K hints/sec** (avg)
- **Kernel Time:** **411.2 ms** median per batch of 219K hints
- **Subset Gen Time:** **0s** (computed on GPU)
- **Data Transfer:** **0 GB** (no subset data needed)

## Production Run Details

```text
======================================================================
RESULTS - 50× H200 GPU PRF Hint Generation (λ=128)
======================================================================
Run ID:              gpu_prf_20260125_205037

Database Parameters:
  n (entries):       1,834,095,877
  Entry size:        40 bytes (5 × int64)
  Database size:     73.4 GB

RMS24 Parameters:
  Block size:        42,826
  Num blocks:        42,828
  Lambda (λ):        128
  Kernel:            hint_gen_with_prf_kernel (ChaCha12 on GPU)

Hint Parameters:
  Total hints:       10,966,016
  Hints per GPU:     219,320

Timing:
  Wall clock time:   4.5 min (271.8s)
  Avg kernel time:   411.2 ms (per 219K hints)
  DB load time:      65-99 sec (per GPU)
  Kernel compile:    ~43 sec (per GPU)
  Subset gen time:   0s (GPU PRF)

Throughput:
  Per-GPU (Avg):     533,500 hints/sec
  Cluster Total:     26,665,695 hints/sec
======================================================================
```

## Comparison: GPU PRF vs CPU Subset (Forge)

| Metric | CPU Subset (Forge) | GPU PRF | Trade-off |
|--------|-------------------|---------|-----------|
| Kernel time (219K hints) | 222 ms | 411 ms | 1.85× slower |
| Per-GPU throughput | 985K/sec | 533K/sec | 1.85× slower |
| Combined (50 GPU) | 49.2M/sec | 26.7M/sec | 1.85× slower |
| Subset gen time | 65s | **0s** | **∞ faster** |
| Data transfer | 37.6 GB/GPU | **0** | **∞ faster** |
| Wall clock | 4.6 min | **4.5 min** | **2% faster** |
| Per-client latency (warm) | 75s | **0.41s** | **183× faster** |

## When to Use Each Approach

### GPU PRF (This Kernel)
- Single-client latency critical (0.41s vs 75s per-client)
- Memory-constrained GPUs (no 37.6GB buffer needed)
- Warm GPU pool (skip subset gen on each request)
- Simpler deployment (no CPU-GPU coordination)

### CPU Subset (Forge Kernel)
- Maximum raw throughput needed
- Batching many clients (amortize 65s subset gen cost)
- CPU cores available for parallel subset generation
- Subset data can be cached/reused

## Cost Analysis

**Modal H200 pricing:** $0.001261/sec GPU, $0.0000131/core/sec CPU

| Phase | CPU Subset | GPU PRF | Reusable? |
|-------|-----------|---------|-----------|
| DB load | 85s | 85s | Yes |
| Kernel compile | 42s | 42s | Yes |
| Subset generation | 65s | **0s** | No |
| Data transfer | ~10s | **0s** | No |
| Kernel execution | 0.22s | 0.41s | No |
| **Total per client (warm)** | **75s** | **0.41s** | |
| **Total cold start** | **192s** | **128s** | |

**Cost per client (warm GPU):**
- CPU Subset: 75s × $0.001261 = **$0.095**
- GPU PRF: 0.41s × $0.001261 = **$0.0005** (190× cheaper)

## Kernel Implementation

The `hint_gen_with_prf_kernel` computes ChaCha12 PRF on-the-fly:

```cuda
// Each block (256 threads) processes one hint
for (int block = tid; block < num_blocks; block += num_threads) {
    uint32_t sel = chacha_prf_select(key, hint_id, block);
    
    if (sel < cutoff) {  // Approximate median: 0x80000000
        uint64_t off = chacha_prf_offset(key, hint_id, block) % block_size;
        int64_t entry_idx = (int64_t)block * block_size + off;
        // XOR entry into local parity
        local_parity[0] ^= entry[0];
        // ...
    }
}
// Warp-level reduction + shared memory reduction
```

## Scripts

- **Benchmark:** `modal run scripts/modal_forge_gpu_subset.py --num-gpus 50 --lambda-param 128`
- **PRF Test:** `modal run scripts/modal_test_gpu_prf.py`

## Artifacts

- [modal_forge_gpu_subset.py](../scripts/modal_forge_gpu_subset.py) - Benchmark script
- [modal_test_gpu_prf.py](../scripts/modal_test_gpu_prf.py) - PRF correctness test
- [subset_gen_kernel.cu](../cuda/subset_gen_kernel.cu) - Standalone subset generation kernel
