#!/usr/bin/env python3
"""
Benchmark comparison: Forge-optimized vs original kernels.

Usage:
    python benchmark_kernels.py                    # Local GPU
    modal run benchmark_kernels.py                 # Modal H200
    modal run benchmark_kernels.py --num-hints 1000 --num-entries 1000000
"""

import time
import argparse
import torch
import torch.nn as nn


def get_pytorch_reference():
    """PyTorch reference implementation."""
    class PytorchModel(nn.Module):
        def forward(self, entries, padded_indices, valid_mask):
            gathered = entries[padded_indices]
            gathered = gathered * valid_mask.unsqueeze(-1).to(gathered.dtype)
            parity = gathered[:, 0, :].clone()
            for i in range(1, gathered.shape[1]):
                parity = torch.bitwise_xor(parity, gathered[:, i, :])
            return parity
    return PytorchModel()


def get_forge_optimized():
    """Forge-optimized CUDA kernel (256 threads, 8 warps)."""
    from forge_optimized import CUDAModel
    return CUDAModel()


def get_minimal_kernel():
    """Original minimal PyTorch kernel."""
    from forge_minimal import HintGenKernel
    return HintGenKernel()


def benchmark_kernel(model, entries, indices, mask, warmup=3, iterations=10):
    """Benchmark a kernel, return median time in ms."""
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
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    times.sort()
    return times[len(times) // 2]  # median


def verify_correctness(model, ref_model, entries, indices, mask):
    """Check that model output matches reference."""
    with torch.no_grad():
        ref_out = ref_model(entries, indices, mask)
        model_out = model(entries, indices, mask)
    
    if torch.equal(ref_out, model_out):
        return True, None
    else:
        diff = (ref_out != model_out).sum().item()
        return False, f"{diff} elements differ"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-entries", type=int, default=262144, help="Database size")
    parser.add_argument("--num-hints", type=int, default=100, help="Number of hints")
    parser.add_argument("--subset-size", type=int, default=512, help="Max subset size per hint")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print("CUDA not available. Exiting.")
        return
    
    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Generate test data
    print(f"Generating test data: {args.num_entries} entries, {args.num_hints} hints, {args.subset_size} subset size")
    torch.manual_seed(42)
    
    entries = torch.randint(0, 2**60, (args.num_entries, 5), dtype=torch.int64, device=device)
    indices = torch.randint(0, args.num_entries, (args.num_hints, args.subset_size), dtype=torch.int64, device=device)
    mask = torch.rand(args.num_hints, args.subset_size, device=device) < 0.8
    
    print(f"Entries: {entries.numel() * 8 / 1e6:.1f} MB")
    print()
    
    # Load models
    print("Loading kernels...")
    ref_model = get_pytorch_reference().to(device)
    forge_model = get_forge_optimized().to(device)
    minimal_model = get_minimal_kernel().to(device)
    
    results = []
    
    # Benchmark PyTorch reference
    print("\n[PyTorch Reference]")
    ref_time = benchmark_kernel(ref_model, entries, indices, mask, args.warmup, args.iterations)
    print(f"  Time: {ref_time:.3f} ms")
    print(f"  Throughput: {args.num_hints / (ref_time / 1000):.0f} hints/sec")
    results.append(("PyTorch Reference", ref_time))
    
    # Benchmark minimal kernel
    print("\n[Minimal Kernel (forge_minimal.py)]")
    minimal_time = benchmark_kernel(minimal_model, entries, indices, mask, args.warmup, args.iterations)
    correct, err = verify_correctness(minimal_model, ref_model, entries, indices, mask)
    print(f"  Time: {minimal_time:.3f} ms")
    print(f"  Throughput: {args.num_hints / (minimal_time / 1000):.0f} hints/sec")
    print(f"  Speedup vs PyTorch: {ref_time / minimal_time:.2f}x")
    print(f"  Correctness: {'PASS' if correct else 'FAIL - ' + err}")
    results.append(("Minimal", minimal_time))
    
    # Benchmark Forge-optimized
    print("\n[Forge-Optimized CUDA Kernel]")
    forge_time = benchmark_kernel(forge_model, entries, indices, mask, args.warmup, args.iterations)
    correct, err = verify_correctness(forge_model, ref_model, entries, indices, mask)
    print(f"  Time: {forge_time:.3f} ms")
    print(f"  Throughput: {args.num_hints / (forge_time / 1000):.0f} hints/sec")
    print(f"  Speedup vs PyTorch: {ref_time / forge_time:.2f}x")
    print(f"  Speedup vs Minimal: {minimal_time / forge_time:.2f}x")
    print(f"  Correctness: {'PASS' if correct else 'FAIL - ' + err}")
    results.append(("Forge-Optimized", forge_time))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Kernel':<25} {'Time (ms)':<12} {'Hints/sec':<12} {'vs PyTorch'}")
    print("-"*60)
    for name, t in results:
        throughput = args.num_hints / (t / 1000)
        speedup = ref_time / t
        print(f"{name:<25} {t:<12.3f} {throughput:<12.0f} {speedup:.2f}x")


if __name__ == "__main__":
    main()
