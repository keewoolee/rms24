/**
 * RMS24 GPU Hint Generation Kernel
 *
 * Adapted from Plinko's hint_kernel.cu for RMS24 protocol.
 * Key difference: Uses median-cutoff subset selection instead of iPRF.
 */

#include <cstdint>
#include <cuda_runtime.h>

#define ENTRY_SIZE 40
#define PARITY_SIZE 48  // 40B aligned to 48B for vectorized loads
#define WARP_SIZE 32

/// RMS24 parameters for GPU kernel
struct Rms24Params {
    uint64_t num_entries;
    uint64_t block_size;
    uint64_t num_blocks;
    uint32_t num_reg_hints;
    uint32_t num_backup_hints;
    uint32_t total_hints;
};

/// Precomputed hint metadata (from CPU Phase 1)
struct HintMeta {
    uint32_t cutoff;
    uint32_t extra_block;
    uint32_t extra_offset;
    uint32_t _padding;
};

/// Output parity
struct HintOutput {
    uint8_t parity[PARITY_SIZE];
};

// ============================================================================
// SHA-256 for PRF (from Plinko)
// ============================================================================

__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __constant__ uint32_t SHA256_H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

/// HMAC-SHA256 based PRF for select/offset
/// Returns 64 bits: high 32 = select value, low 32 = offset value
__device__ uint64_t hmac_prf(
    const uint32_t key[8],
    uint32_t hint_id,
    uint32_t block,
    bool is_offset
) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = SHA256_H0[i];

    // Build message: key || prefix || hint_id || block || padding
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = key[i];
    
    // Prefix: "select" or "offset" (6 bytes each)
    if (is_offset) {
        w[8] = 0x6f666673;  // "offs"
        w[9] = 0x65740000;  // "et\0\0"
    } else {
        w[8] = 0x73656c65;  // "sele"
        w[9] = 0x63740000;  // "ct\0\0"
    }
    w[10] = hint_id;
    w[11] = block;
    w[12] = 0x80000000;  // padding
    w[13] = 0;
    w[14] = 0;
    w[15] = 48 * 8;  // message length in bits (48 bytes)

    // Extend
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }

    // Compress
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e,f,g) + SHA256_K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Return first 64 bits
    return ((uint64_t)(state[0] + a) << 32) | (state[1] + b);
}

// ============================================================================
// Main Hint Generation Kernel
// ============================================================================

/**
 * Generate hint parities using GPU parallelism.
 * 
 * Each thread handles one hint. Streams through all blocks,
 * checking subset membership and XORing entries.
 *
 * Phase 1 (cutoffs, extras) is done on CPU. This kernel does Phase 2 only.
 */
extern "C" __global__ void rms24_hint_gen_kernel(
    const Rms24Params params,
    const uint32_t* __restrict__ prf_key,  // 8 u32s
    const HintMeta* __restrict__ hint_meta,
    const uint8_t* __restrict__ entries,
    HintOutput* __restrict__ output,
    HintOutput* __restrict__ backup_high_output  // For backup hints only
) {
    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    const HintMeta& meta = hint_meta[hint_idx];
    if (meta.cutoff == 0) {
        // Invalid hint, zero output
        uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
        #pragma unroll
        for (int i = 0; i < 6; i++) out_ptr[i] = 0;
        return;
    }

    // Load PRF key into registers
    uint32_t key[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) key[i] = prf_key[i];

    uint64_t parity[5] = {0, 0, 0, 0, 0};  // 40 bytes
    uint64_t parity_high[5] = {0, 0, 0, 0, 0};  // For backup hints
    bool is_regular = hint_idx < params.num_reg_hints;

    for (uint64_t block = 0; block < params.num_blocks; block++) {
        // Compute PRF values
        uint64_t select_prf = hmac_prf(key, hint_idx, (uint32_t)block, false);
        uint64_t offset_prf = hmac_prf(key, hint_idx, (uint32_t)block, true);
        
        uint32_t select_value = (uint32_t)(select_prf >> 32);
        uint64_t picked_offset = offset_prf % params.block_size;
        uint64_t entry_idx = block * params.block_size + picked_offset;

        if (entry_idx >= params.num_entries) continue;

        bool is_selected = select_value < meta.cutoff;

        if (is_regular) {
            if (is_selected) {
                // XOR entry into parity
                const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
                parity[0] ^= entry_ptr[0];
                parity[1] ^= entry_ptr[1];
                parity[2] ^= entry_ptr[2];
                parity[3] ^= entry_ptr[3];
                parity[4] ^= entry_ptr[4];
            } else if (block == meta.extra_block) {
                // XOR extra entry
                uint64_t extra_idx = block * params.block_size + meta.extra_offset;
                if (extra_idx < params.num_entries) {
                    const uint64_t* entry_ptr = (const uint64_t*)(entries + extra_idx * ENTRY_SIZE);
                    parity[0] ^= entry_ptr[0];
                    parity[1] ^= entry_ptr[1];
                    parity[2] ^= entry_ptr[2];
                    parity[3] ^= entry_ptr[3];
                    parity[4] ^= entry_ptr[4];
                }
            }
        } else {
            // Backup hint: track both low and high parities
            const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
            if (is_selected) {
                parity[0] ^= entry_ptr[0];
                parity[1] ^= entry_ptr[1];
                parity[2] ^= entry_ptr[2];
                parity[3] ^= entry_ptr[3];
                parity[4] ^= entry_ptr[4];
            } else {
                parity_high[0] ^= entry_ptr[0];
                parity_high[1] ^= entry_ptr[1];
                parity_high[2] ^= entry_ptr[2];
                parity_high[3] ^= entry_ptr[3];
                parity_high[4] ^= entry_ptr[4];
            }
        }
    }

    // Write main parity output
    uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
    out_ptr[0] = parity[0];
    out_ptr[1] = parity[1];
    out_ptr[2] = parity[2];
    out_ptr[3] = parity[3];
    out_ptr[4] = parity[4];
    out_ptr[5] = 0;  // Padding

    // Write backup high parity if applicable
    if (!is_regular && backup_high_output != nullptr) {
        uint32_t backup_idx = hint_idx - params.num_reg_hints;
        uint64_t* high_ptr = (uint64_t*)backup_high_output[backup_idx].parity;
        high_ptr[0] = parity_high[0];
        high_ptr[1] = parity_high[1];
        high_ptr[2] = parity_high[2];
        high_ptr[3] = parity_high[3];
        high_ptr[4] = parity_high[4];
        high_ptr[5] = 0;
    }
}
