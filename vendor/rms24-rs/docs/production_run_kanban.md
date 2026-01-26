# RMS24 Production Run Kanban (λ=128)

## Configuration

| Parameter | Value |
|-----------|-------|
| Database | mainnet-v3 (73.4 GB, 1.83B entries) |
| Lambda (λ) | 128 |
| Block size | 42,826 |
| Subset size | 21,413 per hint |
| Regular hints | 5,481,728 |
| Backup hints | 5,481,728 |
| **Total hints** | **10,963,456** |
| GPUs | 50× H200 |
| Hints per GPU | ~219,269 |

## Pre-Run Checklist

- [x] Forge kernel ported to hint_kernel.cu
- [x] Modal benchmark script created (modal_forge_fulldb.py)
- [x] Test run with 50K hints passed
- [x] Fix int32 overflow in subset_starts (need int64)
- [x] Verify CUDA kernel uses int64 for start/size

## Bug Fixes Required

### 1. Int64 for subset indices
```python
# modal_forge_fulldb.py line 229-230
subset_starts_t = torch.tensor(subset_starts, dtype=torch.int64, device=device)
subset_sizes_t = torch.tensor(subset_sizes_list, dtype=torch.int64, device=device)
```

### 2. CUDA kernel signature
```cuda
// Change int32_t -> int64_t for starts/sizes
const int64_t* __restrict__ subset_starts,
const int64_t* __restrict__ subset_sizes,
```

**Status:** Completed successfully on 2026-01-25.

## Run Command

```bash
modal run scripts/modal_forge_fulldb.py --num-gpus 50 --lambda-param 128
```

## Actual Results (2026-01-25)

| Metric | Result |
|--------|--------|
| Total hints | 10,966,016 |
| Per-GPU hints | 219,320 |
| Kernel time per GPU | **222.7 ms** (median) |
| DB load time | 66-113s |
| Compile time | ~43s |
| **Wall clock** | **5.4 min (323.5s)** |
| Combined throughput | **49.2M hints/sec** |

## Post-Run Tasks

- [x] Save JSON results to docs/benchmark_forge_production_lambda128_20260125.json
- [x] Update gpu_hint_benchmark_2026-01-25_forge.md with production numbers
- [x] Update KANBAN.md with final results
- [x] Compare with Plinko (33.5M hints in 12.7 min) - RMS24 is **2.4× faster**

## Comparison: RMS24 vs Plinko

| Metric | Plinko (λ=128) | RMS24 (λ=128) |
|--------|----------------|---------------|
| Total hints | 33,554,432 | 10,966,016 |
| Hint formula | 2 × λ × w | 2 × λ × √n |
| Per-GPU throughput | 2,660/sec | **985,000/sec** |
| Compute time | 4.2 min | **~0.2 sec** |
| Wall clock | 12.7 min | **5.4 min** |

## Notes

- RMS24 has fewer hints because block_size = √n ≈ 42K, not w = 131K
- RMS24 kernel is **370× faster** because subsets are precomputed (no SwapOrNot on GPU)
- Main bottleneck is DB loading (66-113s) and subset data transfer (37.6 GB per GPU)
