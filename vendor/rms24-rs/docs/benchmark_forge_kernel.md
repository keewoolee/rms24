# RMS24 Forge-Optimized Kernel Benchmark Results

**Date:** 2026-01-25  
**GPUs:** 50x NVIDIA H200 (150GB each)  
**Platform:** Modal.com

## Summary

| Metric | Value |
|--------|-------|
| **Full DB Combined Throughput** | **38.4 million hints/sec** |
| Per-GPU Throughput | 768K hints/sec |
| Database Size | 73.4 GB (1.83B entries) |
| Subset Size | 21,413 entries/hint |
| Kernel Time | 1.30 ms median |
| All GPUs | 50/50 successful |

## Full Database Benchmark (mainnet-v3)

| GPU | Hints | Median ms | Hints/sec |
|-----|-------|-----------|-----------|
| Best (GPU 49) | 1,000 | 1.283 | 779,305 |
| Worst (GPU 6) | 1,000 | 1.341 | 745,800 |
| Average | 1,000 | 1.302 | 768,043 |
| **50 GPUs Combined** | **50,000** | - | **38,402,130** |

### Key Metrics
- **Database:** 73.4 GB, 1,834,095,877 entries
- **Block size:** 42,826 (sqrt of entries)
- **Subset size:** 21,413 per hint (half of blocks)
- **DB load time:** 70-141 seconds per GPU
- **Kernel compile:** ~43 seconds

## Synthetic Benchmark (S=1024)

| Metric | Forge (256 threads) | Warp (32 threads) | Winner |
|--------|---------------------|-------------------|--------|
| Threads per hint | 256 (8 warps) | 32 (1 warp) | - |
| Correctness | PASS | PASS | - |
| Small subsets (S=1024) | 0.094 ms | 0.088 ms | Warp 1.07x |
| Combined throughput (50 GPU) | 1.06B hints/sec | 1.14B hints/sec | Warp |

## Test Configurations

### Synthetic Benchmark (S=1024)
- **Entries:** 10,000,000
- **Hints:** 100,000 total (2,000 per GPU)
- **Subset size:** 1,024 entries per hint
- **Result:** Warp kernel 7% faster

This is expected because with small subsets (1024 entries), the 32-thread warp kernel has enough parallelism. The 256-thread Forge kernel has overhead from shared memory synchronization.

### Single GPU Results (H200)

From earlier Modal benchmark (`modal_benchmark.py`):

| Config | Forge ms | Throughput | vs PyTorch |
|--------|----------|------------|------------|
| 262K entries, 100 hints, S=512 | 0.025 | 4.0M hints/sec | 261x |
| 1M entries, 100 hints, S=512 | 0.024 | 4.1M hints/sec | 252x |
| 1M entries, 1K hints, S=512 | 0.028 | 35.3M hints/sec | 193x |
| 10M entries, 100 hints, S=1024 | 0.023 | 4.3M hints/sec | 470x |
| 10M entries, 1K hints, S=1024 | 0.053 | 18.9M hints/sec | 231x |

## When to Use Which Kernel

| Subset Size | Recommended Kernel | Reason |
|-------------|-------------------|--------|
| S < 256 | Warp (32 threads) | Not enough work for 256 threads |
| S = 256-1024 | Either (similar) | Both perform well |
| S > 1024 | Forge (256 threads) | Better thread utilization |
| S > 4096 | Forge (256 threads) | Much better parallelism |

## Full Database Estimates

For mainnet-v3 database:
- **Size:** 73 GB (1.8B entries)
- **Block size:** ~42,500 (sqrt of entries)
- **Subset size:** ~21,250 per hint (half of blocks)

At this scale, Forge kernel should significantly outperform Warp kernel due to:
1. 8x more threads processing 21K entries in parallel
2. Better memory coalescing with strided access
3. Amortized reduction overhead

**Estimated throughput:** 500K-1M hints/sec per H200

## Kernel Implementation Details

### Forge Kernel (`rms24_hint_gen_forge_v2_kernel`)
```
Launch: grid=(num_hints), block=(256)
- 256 threads / 8 warps per hint
- Strided iteration over subset
- Two-level reduction: warp shuffle + shared memory
- Single thread final output
```

### Warp Kernel (`rms24_hint_gen_warp_kernel`)
```
Launch: grid=(num_hints), block=(32)
- 32 threads / 1 warp per hint
- Strided iteration over subset  
- Single-level warp shuffle reduction
- Lane 0 writes output
```

## Files

- [hint_kernel.cu](../cuda/hint_kernel.cu) - All kernel implementations
- [modal_forge_bench.py](../scripts/modal_forge_bench.py) - Synthetic benchmark script
- [modal_forge_fulldb.py](../scripts/modal_forge_fulldb.py) - Full database benchmark
- [benchmark_forge_50gpu_synthetic_20260125.json](benchmark_forge_50gpu_synthetic_20260125.json) - Synthetic results
- [benchmark_forge_fulldb_50gpu_20260125.json](benchmark_forge_fulldb_50gpu_20260125.json) - Full DB results

## Previous Benchmarks

From KANBAN.md (before Forge optimization):
- Old kernel (PRF on GPU): 98 hints/sec
- Warp kernel: 854-2,042 hints/sec  
- 50x H200 distributed: 22,855 hints/sec

Current (post-Forge, full DB):
- Single GPU: 768K hints/sec (73GB DB, S=21K)
- 50x H200: **38.4M hints/sec combined**

**Improvement over original:** 
- vs old kernel: 7,800x faster (98 -> 768K per GPU)
- vs distributed warp: 1,680x faster (22.8K -> 38.4M combined)
