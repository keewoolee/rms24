#!/usr/bin/env python3
"""
Test GPU PRF kernel correctness against reference implementation.

Verifies that ChaCha12 PRF on GPU produces same results as CPU.
"""

import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

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

// Kernel to test ChaCha12 PRF values
__global__ void test_prf_kernel(
    const uint32_t* __restrict__ prf_key,
    uint32_t hint_id,
    uint32_t num_blocks,
    uint32_t* __restrict__ select_out,
    uint64_t* __restrict__ offset_out
) {
    uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= num_blocks) return;

    // Load key
    uint32_t key[8];
    for (int i = 0; i < 8; i++) key[i] = prf_key[i];

    // Compute select (domain=0)
    uint32_t sel_output[16];
    chacha12_block(key, 0, hint_id, block, sel_output);
    select_out[block] = sel_output[0];

    // Compute offset (domain=1)
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
    """Python ChaCha12 quarter round."""
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
    """Python ChaCha12 block function."""
    state = [
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7],
        0, nonce0, nonce1, nonce2
    ]
    initial = state.copy()

    for _ in range(6):  # 12 rounds = 6 double-rounds
        # Column round
        state[0], state[4], state[8], state[12] = chacha_quarter_round(state[0], state[4], state[8], state[12])
        state[1], state[5], state[9], state[13] = chacha_quarter_round(state[1], state[5], state[9], state[13])
        state[2], state[6], state[10], state[14] = chacha_quarter_round(state[2], state[6], state[10], state[14])
        state[3], state[7], state[11], state[15] = chacha_quarter_round(state[3], state[7], state[11], state[15])
        # Diagonal round
        state[0], state[5], state[10], state[15] = chacha_quarter_round(state[0], state[5], state[10], state[15])
        state[1], state[6], state[11], state[12] = chacha_quarter_round(state[1], state[6], state[11], state[12])
        state[2], state[7], state[8], state[13] = chacha_quarter_round(state[2], state[7], state[8], state[13])
        state[3], state[4], state[9], state[14] = chacha_quarter_round(state[3], state[4], state[9], state[14])

    return [(state[i] + initial[i]) & 0xFFFFFFFF for i in range(16)]


def prf_select_cpu(key, hint_id, block):
    """CPU PRF select (domain=0)."""
    output = chacha12_block_cpu(key, 0, hint_id, block)
    return output[0]


def prf_offset_cpu(key, hint_id, block):
    """CPU PRF offset (domain=1)."""
    output = chacha12_block_cpu(key, 1, hint_id, block)
    return (output[1] << 32) | output[0]


def main():
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

    # Test key (8 x u32)
    key = [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
           0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c]
    key_t = torch.tensor(key, dtype=torch.int32)

    hint_id = 42
    num_blocks = 1000

    print(f"\nTesting PRF with hint_id={hint_id}, num_blocks={num_blocks}")

    # GPU computation
    select_gpu, offset_gpu = cuda_module.test_prf(key_t, hint_id, num_blocks)
    select_gpu = select_gpu.cpu().numpy().astype(np.uint32)
    offset_gpu = offset_gpu.cpu().numpy().astype(np.uint64)

    # CPU reference
    select_cpu = np.array([prf_select_cpu(key, hint_id, b) for b in range(num_blocks)], dtype=np.uint32)
    offset_cpu = np.array([prf_offset_cpu(key, hint_id, b) for b in range(num_blocks)], dtype=np.uint64)

    # Compare
    select_match = np.array_equal(select_gpu, select_cpu)
    offset_match = np.array_equal(offset_gpu, offset_cpu)

    print(f"\nSelect values match: {select_match}")
    print(f"Offset values match: {offset_match}")

    if not select_match:
        mismatches = np.where(select_gpu != select_cpu)[0][:5]
        print(f"First select mismatches at blocks: {mismatches}")
        for b in mismatches:
            print(f"  Block {b}: GPU={select_gpu[b]:08x}, CPU={select_cpu[b]:08x}")

    if not offset_match:
        mismatches = np.where(offset_gpu != offset_cpu)[0][:5]
        print(f"First offset mismatches at blocks: {mismatches}")
        for b in mismatches:
            print(f"  Block {b}: GPU={offset_gpu[b]:016x}, CPU={offset_cpu[b]:016x}")

    if select_match and offset_match:
        print("\n[OK] GPU PRF matches CPU reference!")
        return 0
    else:
        print("\n[FAIL] GPU PRF does not match CPU reference")
        return 1


if __name__ == "__main__":
    exit(main())
