#!/usr/bin/env python3
"""
Forge kernel with GPU-based subset generation.

Eliminates the 65s CPU subset generation and 37.6GB data transfer by
generating subset indices directly on GPU using ChaCha12 PRF.

Usage:
    modal run scripts/modal_forge_gpu_subset.py
    modal run scripts/modal_forge_gpu_subset.py --num-gpus 50 --lambda-param 128
"""

import modal
import time
import json
from datetime import datetime

app = modal.App("rms24-gpu-subset")
volume = modal.Volume.from_name("plinko-data", create_if_missing=True)

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .pip_install("ninja", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0"})
)

# Combined kernel: subset generation + parity computation
COMBINED_KERNEL_SOURCE = '''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define ENTRY_SIZE_U64 5
#define CHACHA_ROUNDS 12

// ============================================================================
// ChaCha12 Implementation
// ============================================================================

__device__ __forceinline__ void chacha_quarter_round(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d
) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8) | (d >> 24);
    c += d; b ^= c; b = (b << 7) | (b >> 25);
}

__device__ void chacha12_block(
    const uint32_t key[8],
    uint32_t nonce0, uint32_t nonce1, uint32_t nonce2,
    uint32_t output[16]
) {
    uint32_t state[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7],
        0, nonce0, nonce1, nonce2
    };

    uint32_t initial[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) initial[i] = state[i];

    #pragma unroll
    for (int i = 0; i < CHACHA_ROUNDS / 2; i++) {
        chacha_quarter_round(state[0], state[4], state[8],  state[12]);
        chacha_quarter_round(state[1], state[5], state[9],  state[13]);
        chacha_quarter_round(state[2], state[6], state[10], state[14]);
        chacha_quarter_round(state[3], state[7], state[11], state[15]);
        chacha_quarter_round(state[0], state[5], state[10], state[15]);
        chacha_quarter_round(state[1], state[6], state[11], state[12]);
        chacha_quarter_round(state[2], state[7], state[8],  state[13]);
        chacha_quarter_round(state[3], state[4], state[9],  state[14]);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) output[i] = state[i] + initial[i];
}

__device__ __forceinline__ uint32_t chacha_prf_select(
    const uint32_t key[8], uint32_t hint_id, uint32_t block
) {
    uint32_t output[16];
    chacha12_block(key, 0, hint_id, block, output);
    return output[0];
}

__device__ __forceinline__ uint64_t chacha_prf_offset(
    const uint32_t key[8], uint32_t hint_id, uint32_t block
) {
    uint32_t output[16];
    chacha12_block(key, 1, hint_id, block, output);
    return ((uint64_t)output[1] << 32) | output[0];
}

// ============================================================================
// Combined Subset Gen + Parity Computation Kernel
// ============================================================================

/**
 * Generate hint parities using on-the-fly subset computation.
 * 
 * Each block (256 threads) processes one hint:
 * 1. Compute select/offset values using ChaCha12 PRF
 * 2. Filter blocks based on approximate median cutoff
 * 3. XOR filtered entries to compute parity
 *
 * This eliminates the need for CPU subset generation and 37.6GB transfer.
 */
__global__ void hint_gen_with_prf_kernel(
    const int64_t* __restrict__ entries,        // Database [num_entries, 5]
    const uint32_t* __restrict__ prf_key,       // PRF key [8]
    int64_t* __restrict__ parities,             // Output parities [num_hints, 5]
    int64_t num_entries,
    int64_t block_size,
    int32_t num_blocks,
    int32_t num_hints,
    int32_t hint_id_offset                      // For distributed generation
) {
    int hint_idx = blockIdx.x;
    if (hint_idx >= num_hints) return;
    
    int hint_id = hint_idx + hint_id_offset;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load PRF key into shared memory
    __shared__ uint32_t s_key[8];
    if (tid < 8) {
        s_key[tid] = prf_key[tid];
    }
    __syncthreads();

    // Copy to registers
    uint32_t key[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        key[i] = s_key[i];
    }

    // Approximate median cutoff for random u32 values
    // This selects ~50% of blocks (expected subset size = num_blocks/2)
    const uint32_t cutoff = 0x80000000u;

    // Local parity accumulator
    int64_t local_parity[5] = {0, 0, 0, 0, 0};

    // Process blocks in strided fashion
    for (int block = tid; block < num_blocks; block += num_threads) {
        uint32_t sel = chacha_prf_select(key, hint_id, block);
        
        if (sel < cutoff) {
            // Block is in subset - compute offset and XOR entry
            uint64_t off = chacha_prf_offset(key, hint_id, block) % block_size;
            int64_t entry_idx = (int64_t)block * block_size + off;
            
            if (entry_idx >= 0 && entry_idx < num_entries) {
                const int64_t* entry = entries + entry_idx * ENTRY_SIZE_U64;
                local_parity[0] ^= entry[0];
                local_parity[1] ^= entry[1];
                local_parity[2] ^= entry[2];
                local_parity[3] ^= entry[3];
                local_parity[4] ^= entry[4];
            }
        }
    }

    // Warp-level reduction
    __shared__ int64_t shared_parity[5 * 32];  // 32 warps max
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (num_threads + 31) / 32;

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

    // Final reduction by thread 0
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

torch::Tensor prf_hint_gen(
    torch::Tensor entries,      // [num_entries, 5] int64
    torch::Tensor prf_key,      // [8] uint32
    int64_t block_size,
    int32_t num_blocks,
    int32_t num_hints,
    int32_t hint_id_offset
) {
    entries = entries.contiguous();
    prf_key = prf_key.contiguous();
    
    int64_t num_entries = entries.size(0);
    
    auto parities = torch::zeros({num_hints, 5}, entries.options());
    
    hint_gen_with_prf_kernel<<<num_hints, 256>>>(
        entries.data_ptr<int64_t>(),
        reinterpret_cast<const uint32_t*>(prf_key.data_ptr<int32_t>()),
        parities.data_ptr<int64_t>(),
        num_entries,
        block_size,
        num_blocks,
        num_hints,
        hint_id_offset
    );
    
    return parities;
}
'''

CPP_SOURCE = '''
torch::Tensor prf_hint_gen(torch::Tensor, torch::Tensor, int64_t, int32_t, int32_t, int32_t);
'''


@app.function(
    image=cuda_image,
    gpu="H200",
    volumes={"/data": volume},
    timeout=3600,
    memory=196608,
)
def benchmark_gpu(
    gpu_id: int,
    num_gpus: int,
    total_hints: int,
    db_path: str,
    prf_key: list,
    warmup: int = 3,
    iterations: int = 10,
) -> dict:
    """Benchmark with GPU-based PRF subset generation."""
    import torch
    from torch.utils.cpp_extension import load_inline
    import numpy as np
    import os
    import math

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU {gpu_id}: {gpu_name}, {gpu_mem:.1f} GB")

    if not os.path.exists(db_path):
        return {"gpu_id": gpu_id, "error": f"Database not found: {db_path}"}

    db_size = os.path.getsize(db_path)
    entry_size = 40
    num_entries = db_size // entry_size
    block_size = int(math.sqrt(num_entries))
    num_blocks = (num_entries + block_size - 1) // block_size
    if num_blocks % 2 == 1:
        num_blocks += 1

    print(f"GPU {gpu_id}: Loading {db_size / 1e9:.1f} GB database")
    print(f"  Entries: {num_entries:,}, Blocks: {num_blocks:,}, Block size: {block_size:,}")

    load_start = time.time()
    db_mmap = np.memmap(db_path, dtype=np.uint8, mode='r')
    db_array = db_mmap.view(np.int64).reshape(num_entries, 5)
    entries = torch.from_numpy(db_array.copy()).to(device)
    load_time = time.time() - load_start
    print(f"GPU {gpu_id}: Database loaded in {load_time:.1f}s")

    print(f"GPU {gpu_id}: Compiling kernel...")
    compile_start = time.time()
    cuda_module = load_inline(
        name=f'prf_hint_gen_{gpu_id}',
        cpp_sources=CPP_SOURCE,
        cuda_sources=COMBINED_KERNEL_SOURCE,
        functions=['prf_hint_gen'],
        verbose=False,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )
    compile_time = time.time() - compile_start
    print(f"GPU {gpu_id}: Compiled in {compile_time:.1f}s")

    # Divide hints across GPUs
    hints_per_gpu = total_hints // num_gpus
    hint_start = gpu_id * hints_per_gpu
    num_hints = hints_per_gpu if gpu_id < num_gpus - 1 else total_hints - hint_start

    print(f"GPU {gpu_id}: Generating {num_hints} hints (PRF on GPU, no CPU subset gen)")

    # PRF key as tensor
    prf_key_t = torch.tensor(prf_key, dtype=torch.int32, device=device)

    # Warmup
    print(f"GPU {gpu_id}: Warming up...")
    for _ in range(warmup):
        _ = cuda_module.prf_hint_gen(
            entries, prf_key_t, block_size, num_blocks, min(num_hints, 1000), hint_start
        )
    torch.cuda.synchronize()

    # Benchmark
    print(f"GPU {gpu_id}: Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        parities = cuda_module.prf_hint_gen(
            entries, prf_key_t, block_size, num_blocks, num_hints, hint_start
        )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iter {i+1}: {elapsed:.3f} ms")

    times.sort()
    median_ms = times[len(times) // 2]
    min_ms = times[0]
    max_ms = times[-1]

    hints_per_sec = num_hints / (median_ms / 1000)

    print(f"GPU {gpu_id}: Done - {hints_per_sec:,.0f} hints/sec (no subset transfer!)")

    return {
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem,
        "db_size_gb": db_size / 1e9,
        "num_entries": num_entries,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_hints": num_hints,
        "load_time_sec": load_time,
        "compile_time_sec": compile_time,
        "subset_gen_time_sec": 0.0,  # No CPU subset gen!
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "hints_per_sec": hints_per_sec,
    }


@app.local_entrypoint()
def main(
    num_gpus: int = 50,
    num_hints: int = 0,
    lambda_param: int = 128,
    db_path: str = "/data/mainnet-v3/database.bin",
):
    """Benchmark with GPU-based PRF subset generation."""
    import math
    import secrets

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if num_hints == 0:
        db_size_estimate = 73.4e9
        entry_size = 40
        num_entries = int(db_size_estimate / entry_size)
        block_size = int(math.sqrt(num_entries))
        num_hints = 2 * lambda_param * block_size
        print(f"Auto-calculated hints: λ={lambda_param}, block_size={block_size:,}")

    # Generate random PRF key (8 × i32, must be in signed int32 range for PyTorch)
    prf_key = [int.from_bytes(secrets.token_bytes(4), 'little') & 0x7FFFFFFF for _ in range(8)]

    print("=" * 70)
    print("RMS24 GPU SUBSET GENERATION BENCHMARK")
    print("=" * 70)
    print(f"GPUs: {num_gpus}x H200")
    print(f"Lambda: {lambda_param}")
    print(f"Total hints: {num_hints:,}")
    print(f"Database: {db_path}")
    print(f"PRF on GPU: YES (no CPU subset generation)")
    print()

    start_time = time.time()

    futures = []
    for gpu_id in range(num_gpus):
        future = benchmark_gpu.spawn(
            gpu_id=gpu_id,
            num_gpus=num_gpus,
            total_hints=num_hints,
            db_path=db_path,
            prf_key=prf_key,
        )
        futures.append(future)

    results = [f.get() for f in futures]
    total_time = time.time() - start_time

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

        print(f"\nSuccessful GPUs: {len(successful)}")
        print(f"Database size: {db_size:.1f} GB ({successful[0]['num_entries']:,} entries)")
        print(f"Total hints: {total_hints_processed:,}")
        print(f"Avg kernel time: {avg_median_ms:.3f} ms")
        print(f"Combined throughput: {total_hints_per_sec:,.0f} hints/sec")
        print(f"Wall time: {total_time:.1f}s")
        print(f"Subset gen time: 0s (GPU PRF)")
        print(f"Data transfer: 0 GB (no subset data)")

        print("\nPer-GPU results:")
        print(f"{'GPU':<6} {'Hints':<8} {'Median ms':<12} {'Hints/sec':<15}")
        print("-" * 45)
        for r in sorted(successful, key=lambda x: x["gpu_id"])[:10]:
            print(f"{r['gpu_id']:<6} {r['num_hints']:<8} {r['median_ms']:<12.3f} {r['hints_per_sec']:<15,.0f}")
        if len(successful) > 10:
            print(f"... and {len(successful) - 10} more")

    output = {
        "timestamp": timestamp,
        "config": {
            "num_gpus": num_gpus,
            "lambda": lambda_param,
            "total_hints": num_hints,
            "db_path": db_path,
            "prf_on_gpu": True,
        },
        "summary": {
            "successful_gpus": len(successful),
            "failed_gpus": len(failed),
            "total_hints_processed": sum(r["num_hints"] for r in successful) if successful else 0,
            "combined_hints_per_sec": sum(r["hints_per_sec"] for r in successful) if successful else 0,
            "wall_time_sec": total_time,
            "subset_gen_time_sec": 0.0,
            "subset_data_transfer_gb": 0.0,
        },
        "results": results,
    }

    if successful:
        output["summary"]["db_size_gb"] = successful[0]["db_size_gb"]
        output["summary"]["num_entries"] = successful[0]["num_entries"]
        output["summary"]["avg_median_ms"] = sum(r["median_ms"] for r in successful) / len(successful)

    print(f"\n--- JSON ---")
    print(json.dumps(output, indent=2))

    return output
