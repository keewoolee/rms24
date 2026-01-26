#!/usr/bin/env python3
"""
Modal benchmark: Compare Forge-optimized vs original kernels on H200.

Usage:
    modal run modal_benchmark.py
    modal run modal_benchmark.py --num-hints 1000 --num-entries 1000000
    modal run modal_benchmark.py --large  # Full mainnet-scale test
"""

import modal

app = modal.App("rms24-kernel-benchmark")

# H200 image with PyTorch and CUDA toolkit for JIT compilation
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .pip_install("ninja")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)


@app.function(
    image=cuda_image,
    gpu="H200",
    timeout=600,
)
def benchmark_kernels(
    num_entries: int = 262144,
    num_hints: int = 100,
    subset_size: int = 512,
    warmup: int = 5,
    iterations: int = 20,
):
    """Run kernel benchmarks on H200."""
    import time
    import torch
    import torch.nn as nn
    from torch.utils.cpp_extension import load_inline
    
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"\nConfig: {num_entries} entries, {num_hints} hints, {subset_size} subset size")
    
    # Generate test data
    torch.manual_seed(42)
    entries = torch.randint(0, 2**60, (num_entries, 5), dtype=torch.int64, device=device)
    indices = torch.randint(0, num_entries, (num_hints, subset_size), dtype=torch.int64, device=device)
    mask = torch.rand(num_hints, subset_size, device=device) < 0.8
    
    print(f"Entries memory: {entries.numel() * 8 / 1e6:.1f} MB")
    
    # PyTorch reference
    class PytorchModel(nn.Module):
        def forward(self, entries, padded_indices, valid_mask):
            gathered = entries[padded_indices]
            gathered = gathered * valid_mask.unsqueeze(-1).to(gathered.dtype)
            parity = gathered[:, 0, :].clone()
            for i in range(1, gathered.shape[1]):
                parity = torch.bitwise_xor(parity, gathered[:, i, :])
            return parity
    
    # Forge-optimized CUDA kernel
    cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void hint_gen_kernel(
    const int64_t* __restrict__ entries,
    const int64_t* __restrict__ padded_indices,
    const bool* __restrict__ valid_mask,
    int64_t* __restrict__ parities,
    int num_entries,
    int num_hints,
    int max_subset_size
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
    
    const int64_t* hint_indices = padded_indices + hint_idx * max_subset_size;
    const bool* hint_mask = valid_mask + hint_idx * max_subset_size;
    
    for (int i = tid; i < max_subset_size; i += num_threads) {
        if (hint_mask[i]) {
            int64_t entry_idx = hint_indices[i];
            const int64_t* entry = entries + entry_idx * 5;
            local_parity[0] ^= entry[0];
            local_parity[1] ^= entry[1];
            local_parity[2] ^= entry[2];
            local_parity[3] ^= entry[3];
            local_parity[4] ^= entry[4];
        }
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_parity[0] ^= __shfl_xor_sync(0xffffffff, local_parity[0], offset);
        local_parity[1] ^= __shfl_xor_sync(0xffffffff, local_parity[1], offset);
        local_parity[2] ^= __shfl_xor_sync(0xffffffff, local_parity[2], offset);
        local_parity[3] ^= __shfl_xor_sync(0xffffffff, local_parity[3], offset);
        local_parity[4] ^= __shfl_xor_sync(0xffffffff, local_parity[4], offset);
    }
    
    if (lane_id == 0) {
        shared_parity[warp_id * 5 + 0] = local_parity[0];
        shared_parity[warp_id * 5 + 1] = local_parity[1];
        shared_parity[warp_id * 5 + 2] = local_parity[2];
        shared_parity[warp_id * 5 + 3] = local_parity[3];
        shared_parity[warp_id * 5 + 4] = local_parity[4];
    }
    
    __syncthreads();
    
    if (warp_id == 0 && lane_id < 5) {
        int64_t final_val = 0;
        for (int w = 0; w < num_warps; w++) {
            final_val ^= shared_parity[w * 5 + lane_id];
        }
        parities[hint_idx * 5 + lane_id] = final_val;
    }
}

torch::Tensor forward(torch::Tensor entries, torch::Tensor padded_indices, torch::Tensor valid_mask) {
    entries = entries.contiguous();
    padded_indices = padded_indices.contiguous();
    valid_mask = valid_mask.contiguous();
    
    int num_entries = entries.size(0);
    int num_hints = padded_indices.size(0);
    int max_subset_size = padded_indices.size(1);
    
    auto parities = torch::zeros({num_hints, 5}, entries.options());
    
    hint_gen_kernel<<<num_hints, 256>>>(
        entries.data_ptr<int64_t>(),
        padded_indices.data_ptr<int64_t>(),
        valid_mask.data_ptr<bool>(),
        parities.data_ptr<int64_t>(),
        num_entries, num_hints, max_subset_size
    );
    
    return parities;
}
'''
    
    cpp_source = 'torch::Tensor forward(torch::Tensor, torch::Tensor, torch::Tensor);'
    
    print("\nCompiling Forge CUDA kernel...")
    compile_start = time.perf_counter()
    cuda_module = load_inline(
        name='hint_gen_forge',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['forward'],
        verbose=False,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )
    compile_time = time.perf_counter() - compile_start
    print(f"Compilation time: {compile_time:.2f}s")
    
    class ForgeModel(nn.Module):
        def forward(self, entries, indices, mask):
            return cuda_module.forward(entries, indices, mask)
    
    ref_model = PytorchModel()
    forge_model = ForgeModel()
    
    def benchmark(model, name):
        # Warmup
        for _ in range(warmup):
            _ = model(entries, indices, mask)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(entries, indices, mask)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        times.sort()
        median = times[len(times) // 2]
        throughput = num_hints / (median / 1000)
        return median, throughput
    
    # Verify correctness
    print("\nVerifying correctness...")
    ref_out = ref_model(entries, indices, mask)
    forge_out = forge_model(entries, indices, mask)
    correct = torch.equal(ref_out, forge_out)
    print(f"Correctness: {'[PASS]' if correct else '[FAIL]'}")
    
    if not correct:
        diff = (ref_out != forge_out).sum().item()
        print(f"  {diff} elements differ")
    
    # Benchmark
    print("\nBenchmarking...")
    
    ref_time, ref_tp = benchmark(ref_model, "PyTorch")
    print(f"PyTorch Reference:  {ref_time:.3f} ms  ({ref_tp:.0f} hints/sec)")
    
    forge_time, forge_tp = benchmark(forge_model, "Forge")
    print(f"Forge-Optimized:    {forge_time:.3f} ms  ({forge_tp:.0f} hints/sec)")
    
    speedup = ref_time / forge_time
    print(f"\nSpeedup: {speedup:.1f}x")
    
    return {
        "gpu": torch.cuda.get_device_name(0),
        "num_entries": num_entries,
        "num_hints": num_hints,
        "subset_size": subset_size,
        "pytorch_ms": ref_time,
        "forge_ms": forge_time,
        "speedup": speedup,
        "forge_hints_per_sec": forge_tp,
        "correct": correct,
    }


@app.function(
    image=cuda_image,
    gpu="H200",
    timeout=1200,
)
def benchmark_large_scale():
    """Test with larger, more realistic parameters."""
    configs = [
        # (entries, hints, subset_size)
        (262144, 100, 512),      # Default
        (1000000, 100, 512),     # 1M entries
        (1000000, 1000, 512),    # 1M entries, 1K hints
        (10000000, 100, 1024),   # 10M entries
        (10000000, 1000, 1024),  # 10M entries, 1K hints
    ]
    
    results = []
    for num_entries, num_hints, subset_size in configs:
        print(f"\n{'='*60}")
        print(f"Config: {num_entries/1e6:.1f}M entries, {num_hints} hints, S={subset_size}")
        print("="*60)
        
        try:
            result = benchmark_kernels.local(
                num_entries=num_entries,
                num_hints=num_hints,
                subset_size=subset_size,
            )
            results.append(result)
        except Exception as e:
            print(f"Failed: {e}")
            results.append({"error": str(e)})
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Entries':<12} {'Hints':<8} {'Subset':<8} {'Forge ms':<12} {'Hints/sec':<12} {'Speedup'}")
    print("-"*60)
    for r in results:
        if "error" not in r:
            print(f"{r['num_entries']:<12} {r['num_hints']:<8} {r['subset_size']:<8} "
                  f"{r['forge_ms']:<12.3f} {r['forge_hints_per_sec']:<12.0f} {r['speedup']:.1f}x")
    
    return results


@app.local_entrypoint()
def main(
    num_entries: int = 262144,
    num_hints: int = 100,
    subset_size: int = 512,
    large: bool = False,
):
    if large:
        results = benchmark_large_scale.remote()
    else:
        result = benchmark_kernels.remote(
            num_entries=num_entries,
            num_hints=num_hints,
            subset_size=subset_size,
        )
        print(f"\nResult: {result}")
