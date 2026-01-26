#!/usr/bin/env python3
"""
Test GPU PRF correctness on Modal.

Verifies ChaCha12 PRF on GPU matches CPU reference.

Usage:
    modal run scripts/modal_test_gpu_prf.py
"""

import modal

app = modal.App("rms24-test-prf")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.5.1", extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .pip_install("ninja", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0"})
)

CUDA_SOURCE = '''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHACHA_ROUNDS 12

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

__global__ void test_prf_kernel(
    const uint32_t* __restrict__ prf_key,
    uint32_t hint_id,
    uint32_t num_blocks,
    uint32_t* __restrict__ select_out,
    uint64_t* __restrict__ offset_out
) {
    uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= num_blocks) return;

    uint32_t key[8];
    for (int i = 0; i < 8; i++) key[i] = prf_key[i];

    // Select (domain=0)
    uint32_t sel_output[16];
    chacha12_block(key, 0, hint_id, block, sel_output);
    select_out[block] = sel_output[0];

    // Offset (domain=1)
    uint32_t off_output[16];
    chacha12_block(key, 1, hint_id, block, off_output);
    offset_out[block] = ((uint64_t)off_output[1] << 32) | off_output[0];
}

std::tuple<torch::Tensor, torch::Tensor> test_prf(
    torch::Tensor prf_key,
    int32_t hint_id,
    int32_t num_blocks
) {
    prf_key = prf_key.contiguous().to(torch::kCUDA);
    
    auto select_out = torch::zeros({num_blocks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto offset_out = torch::zeros({num_blocks}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    
    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;
    
    test_prf_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint32_t*>(prf_key.data_ptr<int32_t>()),
        hint_id,
        num_blocks,
        reinterpret_cast<uint32_t*>(select_out.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(offset_out.data_ptr<int64_t>())
    );
    
    return std::make_tuple(select_out, offset_out);
}
'''

CPP_SOURCE = '''
std::tuple<torch::Tensor, torch::Tensor> test_prf(torch::Tensor, int32_t, int32_t);
'''


def chacha_quarter_round(a, b, c, d):
    a = (a + b) & 0xFFFFFFFF
    d ^= a
    d = ((d << 16) | (d >> 16)) & 0xFFFFFFFF
    c = (c + d) & 0xFFFFFFFF
    b ^= c
    b = ((b << 12) | (b >> 20)) & 0xFFFFFFFF
    a = (a + b) & 0xFFFFFFFF
    d ^= a
    d = ((d << 8) | (d >> 24)) & 0xFFFFFFFF
    c = (c + d) & 0xFFFFFFFF
    b ^= c
    b = ((b << 7) | (b >> 25)) & 0xFFFFFFFF
    return a, b, c, d


def chacha12_block_cpu(key, nonce0, nonce1, nonce2):
    state = [
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7],
        0, nonce0, nonce1, nonce2
    ]
    initial = state.copy()

    for _ in range(6):
        state[0], state[4], state[8], state[12] = chacha_quarter_round(state[0], state[4], state[8], state[12])
        state[1], state[5], state[9], state[13] = chacha_quarter_round(state[1], state[5], state[9], state[13])
        state[2], state[6], state[10], state[14] = chacha_quarter_round(state[2], state[6], state[10], state[14])
        state[3], state[7], state[11], state[15] = chacha_quarter_round(state[3], state[7], state[11], state[15])
        state[0], state[5], state[10], state[15] = chacha_quarter_round(state[0], state[5], state[10], state[15])
        state[1], state[6], state[11], state[12] = chacha_quarter_round(state[1], state[6], state[11], state[12])
        state[2], state[7], state[8], state[13] = chacha_quarter_round(state[2], state[7], state[8], state[13])
        state[3], state[4], state[9], state[14] = chacha_quarter_round(state[3], state[4], state[9], state[14])

    return [(state[i] + initial[i]) & 0xFFFFFFFF for i in range(16)]


@app.function(image=cuda_image, gpu="H200", timeout=600)
def test_prf_correctness() -> dict:
    """Test GPU PRF correctness against CPU reference."""
    import torch
    from torch.utils.cpp_extension import load_inline
    import numpy as np

    print("Compiling CUDA kernel...")
    cuda_module = load_inline(
        name='test_prf',
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=['test_prf'],
        verbose=False,
        extra_cuda_cflags=['-O3'],
    )
    print("Compiled!")

    results = {"tests": [], "passed": 0, "failed": 0}

    # Test cases
    test_cases = [
        {"key": [0] * 8, "hint_id": 0, "num_blocks": 100},
        {"key": [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
                 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
         "hint_id": 42, "num_blocks": 1000},
        {"key": [0x7FFFFFFF] * 8, "hint_id": 12345, "num_blocks": 500},  # Max signed int32
    ]

    for i, tc in enumerate(test_cases):
        key = tc["key"]
        hint_id = tc["hint_id"]
        num_blocks = tc["num_blocks"]

        key_t = torch.tensor(key, dtype=torch.int32)

        # GPU computation
        select_gpu, offset_gpu = cuda_module.test_prf(key_t, hint_id, num_blocks)
        select_gpu = select_gpu.cpu().numpy().astype(np.uint32)
        offset_gpu = offset_gpu.cpu().numpy().astype(np.uint64)

        # CPU reference
        select_cpu = []
        offset_cpu = []
        for b in range(num_blocks):
            out_sel = chacha12_block_cpu(key, 0, hint_id, b)
            select_cpu.append(out_sel[0])
            out_off = chacha12_block_cpu(key, 1, hint_id, b)
            offset_cpu.append((out_off[1] << 32) | out_off[0])

        select_cpu = np.array(select_cpu, dtype=np.uint32)
        offset_cpu = np.array(offset_cpu, dtype=np.uint64)

        select_match = np.array_equal(select_gpu, select_cpu)
        offset_match = np.array_equal(offset_gpu, offset_cpu)

        test_result = {
            "test_id": i,
            "hint_id": hint_id,
            "num_blocks": num_blocks,
            "select_match": select_match,
            "offset_match": offset_match,
            "passed": select_match and offset_match,
        }

        if not select_match:
            mismatches = np.where(select_gpu != select_cpu)[0][:3]
            test_result["select_mismatches"] = [
                {"block": int(b), "gpu": hex(int(select_gpu[b])), "cpu": hex(int(select_cpu[b]))}
                for b in mismatches
            ]

        if not offset_match:
            mismatches = np.where(offset_gpu != offset_cpu)[0][:3]
            test_result["offset_mismatches"] = [
                {"block": int(b), "gpu": hex(int(offset_gpu[b])), "cpu": hex(int(offset_cpu[b]))}
                for b in mismatches
            ]

        results["tests"].append(test_result)
        if test_result["passed"]:
            results["passed"] += 1
            print(f"[PASS] Test {i}: hint_id={hint_id}, num_blocks={num_blocks}")
        else:
            results["failed"] += 1
            print(f"[FAIL] Test {i}: hint_id={hint_id}, num_blocks={num_blocks}")
            if "select_mismatches" in test_result:
                print(f"       Select mismatches: {test_result['select_mismatches']}")
            if "offset_mismatches" in test_result:
                print(f"       Offset mismatches: {test_result['offset_mismatches']}")

    print(f"\nTotal: {results['passed']} passed, {results['failed']} failed")
    return results


@app.local_entrypoint()
def main():
    """Run PRF correctness tests on Modal."""
    print("=" * 60)
    print("RMS24 GPU PRF CORRECTNESS TEST")
    print("=" * 60)

    results = test_prf_correctness.remote()

    if results["failed"] == 0:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print(f"\n[FAIL] {results['failed']} tests failed")
        return 1
