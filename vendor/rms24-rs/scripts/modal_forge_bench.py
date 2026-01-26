#!/usr/bin/env python3
"""
Forge-optimized RMS24 kernel benchmark on 50x H200 GPUs.

Compares:
- Forge kernel (256 threads, 8 warps per hint)
- Warp kernel (32 threads, 1 warp per hint)

Usage:
    modal run scripts/modal_forge_bench.py
    modal run scripts/modal_forge_bench.py --num-gpus 50 --num-hints 100000
    modal run scripts/modal_forge_bench.py --full-db
"""

import modal
import time
import json
from datetime import datetime

app = modal.App("rms24-forge-bench")
volume = modal.Volume.from_name("plinko-data", create_if_missing=True)

# CUDA devel image for kernel JIT
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .pip_install("ninja", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0"})
)


# Forge-optimized CUDA kernel (256 threads)
FORGE_CUDA_SOURCE = '''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define ENTRY_SIZE_U64 5

__global__ void hint_gen_forge_kernel(
    const int64_t* __restrict__ entries,        // [num_entries, 5]
    const int64_t* __restrict__ subset_indices, // [total_subset_elements]
    const int32_t* __restrict__ subset_starts,  // [num_hints]
    const int32_t* __restrict__ subset_sizes,   // [num_hints]
    int64_t* __restrict__ parities,             // [num_hints, 5]
    int num_entries,
    int num_hints
) {
    int hint_idx = blockIdx.x;
    if (hint_idx >= num_hints) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    __shared__ int64_t shared_parity[5 * 32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (num_threads + 31) / 32;

    int64_t local_parity[5] = {0, 0, 0, 0, 0};

    int start = subset_starts[hint_idx];
    int size = subset_sizes[hint_idx];

    for (int i = tid; i < size; i += num_threads) {
        int64_t entry_idx = subset_indices[start + i];
        if (entry_idx >= 0 && entry_idx < num_entries) {
            const int64_t* entry = entries + entry_idx * ENTRY_SIZE_U64;
            local_parity[0] ^= entry[0];
            local_parity[1] ^= entry[1];
            local_parity[2] ^= entry[2];
            local_parity[3] ^= entry[3];
            local_parity[4] ^= entry[4];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_parity[0] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[0], offset);
        local_parity[1] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[1], offset);
        local_parity[2] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[2], offset);
        local_parity[3] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[3], offset);
        local_parity[4] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[4], offset);
    }

    if (lane_id == 0) {
        for (int i = 0; i < 5; i++) {
            shared_parity[warp_id * 5 + i] = local_parity[i];
        }
    }

    __syncthreads();

    if (tid == 0) {
        int64_t final_parity[5] = {0, 0, 0, 0, 0};
        for (int w = 0; w < num_warps; w++) {
            final_parity[0] ^= shared_parity[w * 5 + 0];
            final_parity[1] ^= shared_parity[w * 5 + 1];
            final_parity[2] ^= shared_parity[w * 5 + 2];
            final_parity[3] ^= shared_parity[w * 5 + 3];
            final_parity[4] ^= shared_parity[w * 5 + 4];
        }

        int64_t* out = parities + hint_idx * 5;
        out[0] = final_parity[0];
        out[1] = final_parity[1];
        out[2] = final_parity[2];
        out[3] = final_parity[3];
        out[4] = final_parity[4];
    }
}

__global__ void hint_gen_warp_kernel(
    const int64_t* __restrict__ entries,
    const int64_t* __restrict__ subset_indices,
    const int32_t* __restrict__ subset_starts,
    const int32_t* __restrict__ subset_sizes,
    int64_t* __restrict__ parities,
    int num_entries,
    int num_hints
) {
    int hint_idx = blockIdx.x;
    if (hint_idx >= num_hints) return;

    int lane = threadIdx.x;
    int64_t local_parity[5] = {0, 0, 0, 0, 0};

    int start = subset_starts[hint_idx];
    int size = subset_sizes[hint_idx];

    for (int i = lane; i < size; i += 32) {
        int64_t entry_idx = subset_indices[start + i];
        if (entry_idx >= 0 && entry_idx < num_entries) {
            const int64_t* entry = entries + entry_idx * ENTRY_SIZE_U64;
            local_parity[0] ^= entry[0];
            local_parity[1] ^= entry[1];
            local_parity[2] ^= entry[2];
            local_parity[3] ^= entry[3];
            local_parity[4] ^= entry[4];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_parity[0] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[0], offset);
        local_parity[1] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[1], offset);
        local_parity[2] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[2], offset);
        local_parity[3] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[3], offset);
        local_parity[4] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[4], offset);
    }

    if (lane == 0) {
        int64_t* out = parities + hint_idx * 5;
        out[0] = local_parity[0];
        out[1] = local_parity[1];
        out[2] = local_parity[2];
        out[3] = local_parity[3];
        out[4] = local_parity[4];
    }
}

torch::Tensor forge_forward(
    torch::Tensor entries,
    torch::Tensor subset_indices,
    torch::Tensor subset_starts,
    torch::Tensor subset_sizes
) {
    entries = entries.contiguous();
    subset_indices = subset_indices.contiguous();
    subset_starts = subset_starts.contiguous();
    subset_sizes = subset_sizes.contiguous();

    int num_entries = entries.size(0);
    int num_hints = subset_starts.size(0);

    auto parities = torch::zeros({num_hints, 5}, entries.options());

    hint_gen_forge_kernel<<<num_hints, 256>>>(
        entries.data_ptr<int64_t>(),
        subset_indices.data_ptr<int64_t>(),
        subset_starts.data_ptr<int32_t>(),
        subset_sizes.data_ptr<int32_t>(),
        parities.data_ptr<int64_t>(),
        num_entries, num_hints
    );

    return parities;
}

torch::Tensor warp_forward(
    torch::Tensor entries,
    torch::Tensor subset_indices,
    torch::Tensor subset_starts,
    torch::Tensor subset_sizes
) {
    entries = entries.contiguous();
    subset_indices = subset_indices.contiguous();
    subset_starts = subset_starts.contiguous();
    subset_sizes = subset_sizes.contiguous();

    int num_entries = entries.size(0);
    int num_hints = subset_starts.size(0);

    auto parities = torch::zeros({num_hints, 5}, entries.options());

    hint_gen_warp_kernel<<<num_hints, 32>>>(
        entries.data_ptr<int64_t>(),
        subset_indices.data_ptr<int64_t>(),
        subset_starts.data_ptr<int32_t>(),
        subset_sizes.data_ptr<int32_t>(),
        parities.data_ptr<int64_t>(),
        num_entries, num_hints
    );

    return parities;
}
'''

CPP_SOURCE = '''
torch::Tensor forge_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor warp_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
'''


@app.function(
    image=cuda_image,
    gpu="H200",
    volumes={"/data": volume},
    timeout=1800,
)
def benchmark_single_gpu(
    gpu_id: int,
    num_entries: int,
    num_hints: int,
    subset_size: int,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    """Benchmark Forge vs Warp kernel on single GPU."""
    import torch
    from torch.utils.cpp_extension import load_inline
    import numpy as np

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    # Compile kernels
    compile_start = time.time()
    cuda_module = load_inline(
        name='hint_gen_bench',
        cpp_sources=CPP_SOURCE,
        cuda_sources=FORGE_CUDA_SOURCE,
        functions=['forge_forward', 'warp_forward'],
        verbose=False,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )
    compile_time = time.time() - compile_start

    # Generate synthetic data
    torch.manual_seed(42 + gpu_id)
    entries = torch.randint(0, 2**60, (num_entries, 5), dtype=torch.int64, device=device)

    # Generate subset data (simulating precomputed subsets)
    subset_indices_list = []
    subset_starts = []
    subset_sizes_list = []
    current_start = 0

    for h in range(num_hints):
        # Random subset of entries
        size = min(subset_size, num_entries)
        indices = torch.randint(0, num_entries, (size,), dtype=torch.int64)
        subset_indices_list.append(indices)
        subset_starts.append(current_start)
        subset_sizes_list.append(size)
        current_start += size

    subset_indices = torch.cat(subset_indices_list).to(device)
    subset_starts = torch.tensor(subset_starts, dtype=torch.int32, device=device)
    subset_sizes = torch.tensor(subset_sizes_list, dtype=torch.int32, device=device)

    def benchmark_kernel(forward_fn, name):
        # Warmup
        for _ in range(warmup):
            _ = forward_fn(entries, subset_indices, subset_starts, subset_sizes)
        torch.cuda.synchronize()

        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = forward_fn(entries, subset_indices, subset_starts, subset_sizes)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        times.sort()
        return times[len(times) // 2]

    # Verify correctness
    forge_out = cuda_module.forge_forward(entries, subset_indices, subset_starts, subset_sizes)
    warp_out = cuda_module.warp_forward(entries, subset_indices, subset_starts, subset_sizes)
    correct = torch.equal(forge_out, warp_out)

    # Benchmark
    forge_ms = benchmark_kernel(cuda_module.forge_forward, "Forge")
    warp_ms = benchmark_kernel(cuda_module.warp_forward, "Warp")

    return {
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "num_entries": num_entries,
        "num_hints": num_hints,
        "subset_size": subset_size,
        "forge_ms": forge_ms,
        "warp_ms": warp_ms,
        "forge_hints_per_sec": num_hints / (forge_ms / 1000),
        "warp_hints_per_sec": num_hints / (warp_ms / 1000),
        "speedup": warp_ms / forge_ms,
        "correct": correct,
        "compile_time": compile_time,
    }


@app.function(
    image=cuda_image,
    gpu="H200",
    volumes={"/data": volume},
    timeout=3600,
    memory=131072,  # 128GB RAM for large DB
)
def benchmark_full_db(
    gpu_id: int,
    num_gpus: int,
    total_hints: int,
    db_path: str = "/data/mainnet-v3/database.bin",
    subset_size: int = 21500,  # sqrt(462M) ~ 21500 blocks per hint
) -> dict:
    """Benchmark with full mainnet database."""
    import torch
    from torch.utils.cpp_extension import load_inline
    import os

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    # Check if DB exists
    if not os.path.exists(db_path):
        return {
            "gpu_id": gpu_id,
            "error": f"Database not found: {db_path}",
        }

    # Load database
    db_size = os.path.getsize(db_path)
    entry_size = 40  # 40 bytes per entry
    num_entries = db_size // entry_size

    print(f"GPU {gpu_id}: Loading {db_size / 1e9:.1f} GB database ({num_entries:,} entries)")

    # Memory-map and convert to tensor
    import numpy as np
    db_mmap = np.memmap(db_path, dtype=np.uint8, mode='r')
    
    # Reshape to [num_entries, 5] of int64 (40 bytes = 5 x 8 bytes)
    db_array = db_mmap.view(np.int64).reshape(num_entries, 5)
    entries = torch.from_numpy(db_array.copy()).to(device)

    print(f"GPU {gpu_id}: Database loaded to GPU")

    # Compile kernels
    cuda_module = load_inline(
        name='hint_gen_fulldb',
        cpp_sources=CPP_SOURCE,
        cuda_sources=FORGE_CUDA_SOURCE,
        functions=['forge_forward', 'warp_forward'],
        verbose=False,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )

    # Divide hints among GPUs
    hints_per_gpu = total_hints // num_gpus
    hint_start = gpu_id * hints_per_gpu
    num_hints = hints_per_gpu if gpu_id < num_gpus - 1 else total_hints - hint_start

    # Generate subset data
    torch.manual_seed(42 + gpu_id)
    subset_indices_list = []
    subset_starts = []
    subset_sizes_list = []
    current_start = 0

    for h in range(num_hints):
        size = min(subset_size, num_entries)
        indices = torch.randint(0, num_entries, (size,), dtype=torch.int64)
        subset_indices_list.append(indices)
        subset_starts.append(current_start)
        subset_sizes_list.append(size)
        current_start += size

    subset_indices = torch.cat(subset_indices_list).to(device)
    subset_starts_t = torch.tensor(subset_starts, dtype=torch.int32, device=device)
    subset_sizes_t = torch.tensor(subset_sizes_list, dtype=torch.int32, device=device)

    # Warmup
    for _ in range(3):
        _ = cuda_module.forge_forward(entries, subset_indices, subset_starts_t, subset_sizes_t)
    torch.cuda.synchronize()

    # Benchmark Forge kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        _ = cuda_module.forge_forward(entries, subset_indices, subset_starts_t, subset_sizes_t)
    torch.cuda.synchronize()
    forge_total_ms = (time.perf_counter() - start) * 1000
    forge_ms = forge_total_ms / 5

    return {
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "num_entries": num_entries,
        "num_hints": num_hints,
        "subset_size": subset_size,
        "db_size_gb": db_size / 1e9,
        "forge_ms": forge_ms,
        "forge_hints_per_sec": num_hints / (forge_ms / 1000),
    }


@app.local_entrypoint()
def main(
    num_gpus: int = 50,
    num_hints: int = 100000,
    num_entries: int = 10000000,
    subset_size: int = 1024,
    full_db: bool = False,
):
    """
    Benchmark Forge-optimized kernel on multiple H200 GPUs.
    
    Results saved to docs/benchmark_forge_YYYYMMDD_HHMMSS.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"=== RMS24 FORGE KERNEL BENCHMARK ===")
    print(f"GPUs: {num_gpus}x H200")
    print(f"Hints: {num_hints:,}")
    print(f"Entries: {num_entries:,}")
    print(f"Subset size: {subset_size}")
    print(f"Full DB: {full_db}")
    print()

    start_time = time.time()

    if full_db:
        # Full database benchmark
        futures = []
        for gpu_id in range(num_gpus):
            future = benchmark_full_db.spawn(
                gpu_id=gpu_id,
                num_gpus=num_gpus,
                total_hints=num_hints,
            )
            futures.append(future)

        results = [f.get() for f in futures]
    else:
        # Synthetic data benchmark
        hints_per_gpu = num_hints // num_gpus

        futures = []
        for gpu_id in range(num_gpus):
            future = benchmark_single_gpu.spawn(
                gpu_id=gpu_id,
                num_entries=num_entries,
                num_hints=hints_per_gpu,
                subset_size=subset_size,
            )
            futures.append(future)

        results = [f.get() for f in futures]

    total_time = time.time() - start_time

    # Aggregate results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if failed:
        print(f"\n{len(failed)} GPUs failed:")
        for r in failed[:5]:
            print(f"  GPU {r['gpu_id']}: {r.get('error', 'unknown')[:100]}")

    if successful:
        total_hints = sum(r["num_hints"] for r in successful)
        
        if not full_db:
            avg_forge_ms = sum(r["forge_ms"] for r in successful) / len(successful)
            avg_warp_ms = sum(r["warp_ms"] for r in successful) / len(successful)
            total_forge_hints_sec = sum(r["forge_hints_per_sec"] for r in successful)
            total_warp_hints_sec = sum(r["warp_hints_per_sec"] for r in successful)
            avg_speedup = sum(r["speedup"] for r in successful) / len(successful)
            all_correct = all(r["correct"] for r in successful)

            print(f"\nSynthetic benchmark ({len(successful)} GPUs):")
            print(f"  Total hints: {total_hints:,}")
            print(f"  Avg Forge time: {avg_forge_ms:.3f} ms")
            print(f"  Avg Warp time: {avg_warp_ms:.3f} ms")
            print(f"  Forge throughput (combined): {total_forge_hints_sec:,.0f} hints/sec")
            print(f"  Warp throughput (combined): {total_warp_hints_sec:,.0f} hints/sec")
            print(f"  Avg speedup (Forge vs Warp): {avg_speedup:.2f}x")
            print(f"  Correctness: {'[PASS]' if all_correct else '[FAIL]'}")
        else:
            avg_forge_ms = sum(r["forge_ms"] for r in successful) / len(successful)
            total_forge_hints_sec = sum(r["forge_hints_per_sec"] for r in successful)

            print(f"\nFull DB benchmark ({len(successful)} GPUs):")
            print(f"  Total hints: {total_hints:,}")
            print(f"  DB size: {successful[0].get('db_size_gb', 0):.1f} GB")
            print(f"  Avg Forge time: {avg_forge_ms:.3f} ms")
            print(f"  Forge throughput (combined): {total_forge_hints_sec:,.0f} hints/sec")

    print(f"\nWall time: {total_time:.2f}s")

    # Save results
    output = {
        "timestamp": timestamp,
        "config": {
            "num_gpus": num_gpus,
            "num_hints": num_hints,
            "num_entries": num_entries,
            "subset_size": subset_size,
            "full_db": full_db,
        },
        "summary": {
            "successful_gpus": len(successful),
            "failed_gpus": len(failed),
            "total_hints": sum(r["num_hints"] for r in successful) if successful else 0,
            "wall_time_sec": total_time,
        },
        "results": results,
    }

    if successful and not full_db:
        output["summary"]["forge_total_hints_per_sec"] = sum(r["forge_hints_per_sec"] for r in successful)
        output["summary"]["warp_total_hints_per_sec"] = sum(r["warp_hints_per_sec"] for r in successful)
        output["summary"]["avg_speedup"] = sum(r["speedup"] for r in successful) / len(successful)
    elif successful and full_db:
        output["summary"]["forge_total_hints_per_sec"] = sum(r["forge_hints_per_sec"] for r in successful)

    # Print JSON for saving
    print(f"\n--- JSON OUTPUT (save to docs/benchmark_forge_{timestamp}.json) ---")
    print(json.dumps(output, indent=2))

    return output
