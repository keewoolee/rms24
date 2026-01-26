#!/usr/bin/env python3
"""
Forge-optimized kernel benchmark on 50x H200 with FULL mainnet-v3 database.

Each GPU loads the entire 73GB database and processes its share of hints.

Usage:
    modal run scripts/modal_forge_fulldb.py
    modal run scripts/modal_forge_fulldb.py --num-gpus 50 --num-hints 50000
"""

import modal
import time
import json
from datetime import datetime

app = modal.App("rms24-forge-fulldb")
volume = modal.Volume.from_name("plinko-data", create_if_missing=True)

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .pip_install("ninja", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0"})
)

FORGE_CUDA_SOURCE = '''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define ENTRY_SIZE_U64 5

__global__ void hint_gen_forge_kernel(
    const int64_t* __restrict__ entries,
    const int64_t* __restrict__ subset_indices,
    const int64_t* __restrict__ subset_starts,
    const int64_t* __restrict__ subset_sizes,
    int64_t* __restrict__ parities,
    int64_t num_entries,
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

    int64_t start = subset_starts[hint_idx];
    int64_t size = subset_sizes[hint_idx];

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

torch::Tensor forge_forward(
    torch::Tensor entries,
    torch::Tensor subset_indices,
    torch::Tensor subset_starts,
    torch::Tensor subset_sizes,
    int64_t num_entries
) {
    entries = entries.contiguous();
    subset_indices = subset_indices.contiguous();
    subset_starts = subset_starts.contiguous();
    subset_sizes = subset_sizes.contiguous();

    int num_hints = subset_starts.size(0);

    auto parities = torch::zeros({num_hints, 5}, entries.options());

    hint_gen_forge_kernel<<<num_hints, 256>>>(
        entries.data_ptr<int64_t>(),
        subset_indices.data_ptr<int64_t>(),
        subset_starts.data_ptr<int64_t>(),
        subset_sizes.data_ptr<int64_t>(),
        parities.data_ptr<int64_t>(),
        num_entries, num_hints
    );

    return parities;
}
'''

CPP_SOURCE = '''
torch::Tensor forge_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
'''


@app.function(
    image=cuda_image,
    gpu="H200",
    volumes={"/data": volume},
    timeout=3600,
    memory=196608,  # 192GB RAM
)
def benchmark_gpu(
    gpu_id: int,
    num_gpus: int,
    total_hints: int,
    db_path: str,
    warmup: int = 3,
    iterations: int = 10,
) -> dict:
    """Benchmark single GPU with full database."""
    import torch
    from torch.utils.cpp_extension import load_inline
    import numpy as np
    import os
    import math

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU {gpu_id}: {gpu_name}, {gpu_mem:.1f} GB")

    # Check database
    if not os.path.exists(db_path):
        return {"gpu_id": gpu_id, "error": f"Database not found: {db_path}"}

    db_size = os.path.getsize(db_path)
    entry_size = 40
    num_entries = db_size // entry_size
    block_size = int(math.sqrt(num_entries))
    subset_size = block_size // 2  # ~half of blocks per hint

    print(f"GPU {gpu_id}: Loading {db_size / 1e9:.1f} GB database")
    print(f"  Entries: {num_entries:,}")
    print(f"  Block size: {block_size:,}")
    print(f"  Subset size: {subset_size:,}")

    # Load database via memory map
    load_start = time.time()
    db_mmap = np.memmap(db_path, dtype=np.uint8, mode='r')
    db_array = db_mmap.view(np.int64).reshape(num_entries, 5)
    entries = torch.from_numpy(db_array.copy()).to(device)
    load_time = time.time() - load_start
    print(f"GPU {gpu_id}: Database loaded in {load_time:.1f}s")

    # Compile kernel
    print(f"GPU {gpu_id}: Compiling kernel...")
    compile_start = time.time()
    cuda_module = load_inline(
        name=f'hint_gen_fulldb_{gpu_id}',
        cpp_sources=CPP_SOURCE,
        cuda_sources=FORGE_CUDA_SOURCE,
        functions=['forge_forward'],
        verbose=False,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )
    compile_time = time.time() - compile_start
    print(f"GPU {gpu_id}: Compiled in {compile_time:.1f}s")

    # Divide hints
    hints_per_gpu = total_hints // num_gpus
    hint_start = gpu_id * hints_per_gpu
    num_hints = hints_per_gpu if gpu_id < num_gpus - 1 else total_hints - hint_start

    print(f"GPU {gpu_id}: Generating {num_hints} hints with subset_size={subset_size}")

    # Generate subset data - VECTORIZED with batching for large hint counts
    torch.manual_seed(42 + gpu_id)
    gen_start = time.time()
    
    total_indices = num_hints * subset_size
    max_batch_indices = 500_000_000  # ~4GB per batch to leave room for DB
    
    if total_indices <= max_batch_indices:
        # Small enough to generate all at once on GPU
        subset_indices = torch.randint(
            0, num_entries, 
            (num_hints, subset_size), 
            dtype=torch.int64, 
            device=device
        ).flatten()
    else:
        # Generate in batches on CPU, then transfer
        batch_size = max_batch_indices // subset_size
        chunks = []
        for start in range(0, num_hints, batch_size):
            end = min(start + batch_size, num_hints)
            chunk = torch.randint(
                0, num_entries, 
                (end - start, subset_size), 
                dtype=torch.int64
            ).flatten()
            chunks.append(chunk)
        subset_indices = torch.cat(chunks).to(device)
    
    # Generate starts and sizes directly on GPU
    subset_starts_t = torch.arange(
        0, num_hints * subset_size, subset_size, 
        dtype=torch.int64, 
        device=device
    )
    subset_sizes_t = torch.full(
        (num_hints,), subset_size, 
        dtype=torch.int64, 
        device=device
    )
    
    gen_time = time.time() - gen_start
    print(f"GPU {gpu_id}: Subset data: {subset_indices.numel() * 8 / 1e6:.1f} MB (generated in {gen_time:.2f}s)")

    # Warmup
    print(f"GPU {gpu_id}: Warming up...")
    for _ in range(warmup):
        _ = cuda_module.forge_forward(entries, subset_indices, subset_starts_t, subset_sizes_t, num_entries)
    torch.cuda.synchronize()

    # Benchmark
    print(f"GPU {gpu_id}: Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = cuda_module.forge_forward(entries, subset_indices, subset_starts_t, subset_sizes_t, num_entries)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iter {i+1}: {elapsed:.3f} ms")

    times.sort()
    median_ms = times[len(times) // 2]
    min_ms = times[0]
    max_ms = times[-1]

    hints_per_sec = num_hints / (median_ms / 1000)

    print(f"GPU {gpu_id}: Done - {hints_per_sec:,.0f} hints/sec")

    return {
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem,
        "db_size_gb": db_size / 1e9,
        "num_entries": num_entries,
        "block_size": block_size,
        "subset_size": subset_size,
        "num_hints": num_hints,
        "load_time_sec": load_time,
        "compile_time_sec": compile_time,
        "gen_time_sec": gen_time,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "hints_per_sec": hints_per_sec,
    }


@app.local_entrypoint()
def main(
    num_gpus: int = 50,
    num_hints: int = 0,  # 0 = auto-calculate from lambda
    lambda_param: int = 128,
    db_path: str = "/data/mainnet-v3/database.bin",
):
    """
    Benchmark Forge kernel on 50x H200 with full mainnet database.
    """
    import math
    import os
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate production hint count if not specified
    # num_entries ≈ 1.834B, block_size = sqrt(n), total_hints = 2 * lambda * block_size
    if num_hints == 0:
        # Estimate from known DB size
        db_size_estimate = 73.4e9  # 73.4 GB
        entry_size = 40
        num_entries = int(db_size_estimate / entry_size)
        block_size = int(math.sqrt(num_entries))
        num_hints = 2 * lambda_param * block_size
        print(f"Auto-calculated hints: λ={lambda_param}, block_size={block_size:,}")

    print("=" * 70)
    print("RMS24 FORGE KERNEL - FULL DATABASE BENCHMARK (PRODUCTION)")
    print("=" * 70)
    print(f"GPUs: {num_gpus}x H200")
    print(f"Lambda: {lambda_param}")
    print(f"Total hints: {num_hints:,}")
    print(f"Database: {db_path}")
    print()

    start_time = time.time()

    # Launch all GPUs in parallel
    futures = []
    for gpu_id in range(num_gpus):
        future = benchmark_gpu.spawn(
            gpu_id=gpu_id,
            num_gpus=num_gpus,
            total_hints=num_hints,
            db_path=db_path,
        )
        futures.append(future)

    # Collect results
    results = [f.get() for f in futures]

    total_time = time.time() - start_time

    # Process results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if failed:
        print(f"\n{len(failed)} GPUs failed:")
        for r in failed[:5]:
            print(f"  GPU {r['gpu_id']}: {r.get('error', 'unknown')[:100]}")

    if successful:
        total_hints_processed = sum(r["num_hints"] for r in successful)
        total_hints_per_sec = sum(r["hints_per_sec"] for r in successful)
        avg_median_ms = sum(r["median_ms"] for r in successful) / len(successful)
        db_size = successful[0]["db_size_gb"]
        subset_size = successful[0]["subset_size"]

        print(f"\nSuccessful GPUs: {len(successful)}")
        print(f"Database size: {db_size:.1f} GB ({successful[0]['num_entries']:,} entries)")
        print(f"Subset size: {subset_size:,} entries per hint")
        print(f"Total hints: {total_hints_processed:,}")
        print(f"Avg kernel time: {avg_median_ms:.3f} ms")
        print(f"Combined throughput: {total_hints_per_sec:,.0f} hints/sec")
        print(f"Wall time: {total_time:.1f}s")

        # Per-GPU breakdown
        print("\nPer-GPU results:")
        print(f"{'GPU':<6} {'Hints':<8} {'Median ms':<12} {'Hints/sec':<15}")
        print("-" * 45)
        for r in sorted(successful, key=lambda x: x["gpu_id"]):
            print(f"{r['gpu_id']:<6} {r['num_hints']:<8} {r['median_ms']:<12.3f} {r['hints_per_sec']:<15,.0f}")

    # Build output JSON
    output = {
        "timestamp": timestamp,
        "config": {
            "num_gpus": num_gpus,
            "lambda": lambda_param,
            "total_hints": num_hints,
            "db_path": db_path,
        },
        "summary": {
            "successful_gpus": len(successful),
            "failed_gpus": len(failed),
            "total_hints_processed": sum(r["num_hints"] for r in successful) if successful else 0,
            "combined_hints_per_sec": sum(r["hints_per_sec"] for r in successful) if successful else 0,
            "wall_time_sec": total_time,
        },
        "results": results,
    }

    if successful:
        output["summary"]["db_size_gb"] = successful[0]["db_size_gb"]
        output["summary"]["num_entries"] = successful[0]["num_entries"]
        output["summary"]["subset_size"] = successful[0]["subset_size"]
        output["summary"]["avg_median_ms"] = sum(r["median_ms"] for r in successful) / len(successful)

    print(f"\n--- JSON (save to docs/benchmark_forge_fulldb_{timestamp}.json) ---")
    print(json.dumps(output, indent=2))

    return output
