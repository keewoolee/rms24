/**
 * RMS24 GPU Hint Generation Kernel
 *
 * Uses ChaCha12 for PRF (matching CPU implementation).
 * Key difference from Plinko: Uses median-cutoff subset selection instead of iPRF.
 *
 * ChaCha12 advantages:
 * - ARX operations only (no memory lookups like AES)
 * - GPU-friendly (same implementation as CPU)
 * - 12 rounds provides sufficient security
 */

#include <cstdint>
#include <cuda_runtime.h>

#define ENTRY_SIZE 40
#define PARITY_SIZE 48  // 40B aligned to 48B for vectorized loads
#define WARP_SIZE 32
#define CHACHA_ROUNDS 12

/// RMS24 parameters for GPU kernel
struct Rms24Params {
    uint64_t num_entries;
    uint64_t block_size;
    uint64_t num_blocks;
    uint32_t num_reg_hints;
    uint32_t num_backup_hints;
    uint32_t total_hints;
    uint32_t _padding;  // Align to 8 bytes
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

/**
 * ChaCha12 block function.
 *
 * Nonce layout (12 bytes / 3 u32s):
 * - nonce[0]: domain tag (0 = select, 1 = offset)
 * - nonce[1]: hint_id
 * - nonce[2]: block
 */
__device__ void chacha12_block(
    const uint32_t key[8],
    uint32_t nonce0,  // domain
    uint32_t nonce1,  // hint_id
    uint32_t nonce2,  // block
    uint32_t output[16]
) {
    // ChaCha state: constants || key || counter || nonce
    uint32_t state[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,  // "expand 32-byte k"
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7],
        0,       // counter (always 0 for PRF)
        nonce0, nonce1, nonce2
    };

    uint32_t initial[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) initial[i] = state[i];

    // 12 rounds (6 double-rounds)
    #pragma unroll
    for (int i = 0; i < CHACHA_ROUNDS / 2; i++) {
        // Column round
        chacha_quarter_round(state[0], state[4], state[8],  state[12]);
        chacha_quarter_round(state[1], state[5], state[9],  state[13]);
        chacha_quarter_round(state[2], state[6], state[10], state[14]);
        chacha_quarter_round(state[3], state[7], state[11], state[15]);
        // Diagonal round
        chacha_quarter_round(state[0], state[5], state[10], state[15]);
        chacha_quarter_round(state[1], state[6], state[11], state[12]);
        chacha_quarter_round(state[2], state[7], state[8],  state[13]);
        chacha_quarter_round(state[3], state[4], state[9],  state[14]);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) output[i] = state[i] + initial[i];
}

/**
 * ChaCha12-based PRF.
 *
 * Domain separation via nonce[0]:
 * - 0 = select (returns first 32 bits)
 * - 1 = offset (returns first 64 bits)
 */
__device__ uint32_t chacha_prf_select(
    const uint32_t key[8],
    uint32_t hint_id,
    uint32_t block
) {
    uint32_t output[16];
    chacha12_block(key, 0, hint_id, block, output);
    return output[0];
}

__device__ uint64_t chacha_prf_offset(
    const uint32_t key[8],
    uint32_t hint_id,
    uint32_t block
) {
    uint32_t output[16];
    chacha12_block(key, 1, hint_id, block, output);
    return ((uint64_t)output[1] << 32) | output[0];
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
    const uint32_t* __restrict__ prf_key,  // 8 u32s (256-bit ChaCha key)
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
        // Compute PRF values using ChaCha12
        uint32_t select_value = chacha_prf_select(key, hint_idx, (uint32_t)block);
        uint64_t offset_prf = chacha_prf_offset(key, hint_idx, (uint32_t)block);
        
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

// ============================================================================
// Warp-Optimized Hint Generation Kernel (Legacy - 32 threads)
// ============================================================================

/**
 * Warp-level reduction for 5 x uint64_t parity values.
 * Uses butterfly shuffle pattern to reduce across all 32 lanes.
 */
__device__ __forceinline__ void warp_reduce_parity(uint64_t parity[5]) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            uint32_t lo = (uint32_t)parity[i];
            uint32_t hi = (uint32_t)(parity[i] >> 32);
            lo ^= __shfl_xor_sync(0xFFFFFFFF, lo, offset);
            hi ^= __shfl_xor_sync(0xFFFFFFFF, hi, offset);
            parity[i] = ((uint64_t)hi << 32) | lo;
        }
    }
}

/**
 * Warp-optimized hint generation kernel.
 * 
 * Each warp (32 threads) processes one hint cooperatively.
 * Subsets are precomputed on CPU - no PRF calls needed.
 *
 * Launch config: grid=(num_hints, 1, 1), block=(32, 1, 1)
 *
 * Memory layout for subset arrays:
 * - subset_starts[hint_idx]: starting index into subset_blocks/subset_offsets
 * - subset_sizes[hint_idx]: number of blocks in this hint's subset
 * - subset_blocks[start..start+size]: block indices for this hint
 * - subset_offsets[start..start+size]: entry offsets within each block
 */
extern "C" __global__ void rms24_hint_gen_warp_kernel(
    const Rms24Params params,
    const uint32_t* __restrict__ subset_blocks,    // Flattened block indices
    const uint32_t* __restrict__ subset_offsets,   // Flattened offsets
    const uint32_t* __restrict__ subset_starts,    // Start index per hint
    const uint32_t* __restrict__ subset_sizes,     // Number of blocks per hint
    const uint32_t* __restrict__ extra_blocks,     // Extra block per hint (UINT32_MAX if none)
    const uint32_t* __restrict__ extra_offsets,    // Extra offset per hint
    const uint8_t* __restrict__ entries,
    HintOutput* __restrict__ output,
    HintOutput* __restrict__ backup_high_output    // For backup hints only (TODO: not implemented)
) {
    uint32_t hint_idx = blockIdx.x;
    uint32_t lane = threadIdx.x;

    if (hint_idx >= params.total_hints) return;

    uint32_t start = subset_starts[hint_idx];
    uint32_t size = subset_sizes[hint_idx];

    uint64_t parity[5] = {0, 0, 0, 0, 0};

    // Each lane processes a strided subset of blocks
    for (uint32_t i = lane; i < size; i += WARP_SIZE) {
        uint32_t block_idx = subset_blocks[start + i];
        uint32_t offset = subset_offsets[start + i];

        uint64_t entry_idx = (uint64_t)block_idx * params.block_size + offset;
        if (entry_idx >= params.num_entries) continue;

        const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
        parity[0] ^= entry_ptr[0];
        parity[1] ^= entry_ptr[1];
        parity[2] ^= entry_ptr[2];
        parity[3] ^= entry_ptr[3];
        parity[4] ^= entry_ptr[4];
    }

    // Warp reduction: XOR all lanes' parities together
    warp_reduce_parity(parity);

    // Lane 0 handles extra entry and writes output
    if (lane == 0) {
        uint32_t extra_block = extra_blocks[hint_idx];
        if (extra_block != UINT32_MAX) {
            uint32_t extra_offset = extra_offsets[hint_idx];
            uint64_t extra_entry_idx = (uint64_t)extra_block * params.block_size + extra_offset;
            if (extra_entry_idx < params.num_entries) {
                const uint64_t* entry_ptr = (const uint64_t*)(entries + extra_entry_idx * ENTRY_SIZE);
                parity[0] ^= entry_ptr[0];
                parity[1] ^= entry_ptr[1];
                parity[2] ^= entry_ptr[2];
                parity[3] ^= entry_ptr[3];
                parity[4] ^= entry_ptr[4];
            }
        }

        uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
        out_ptr[0] = parity[0];
        out_ptr[1] = parity[1];
        out_ptr[2] = parity[2];
        out_ptr[3] = parity[3];
        out_ptr[4] = parity[4];
        out_ptr[5] = 0;  // Padding
    }

    // TODO: Backup hints need high parity (blocks NOT in subset).
    // For now, backup hints should use the original kernel or
    // precompute both subsets on CPU.
    (void)backup_high_output;
}

// ============================================================================
// Forge-Optimized Hint Generation Kernel (256 threads, 8 warps)
// ============================================================================

/**
 * Forge-optimized hint generation kernel.
 * 
 * Each block (256 threads / 8 warps) processes one hint cooperatively.
 * Two-level reduction: warp shuffle + shared memory across warps.
 *
 * Launch config: grid=(num_hints, 1, 1), block=(256, 1, 1)
 *
 * Key optimizations over warp kernel:
 * - 8x more threads per hint for better occupancy
 * - Strided iteration handles large subsets efficiently
 * - Two-level reduction minimizes synchronization
 */
extern "C" __global__ void rms24_hint_gen_forge_kernel(
    const Rms24Params params,
    const uint32_t* __restrict__ subset_blocks,    // Flattened block indices
    const uint32_t* __restrict__ subset_offsets,   // Flattened offsets
    const uint32_t* __restrict__ subset_starts,    // Start index per hint
    const uint32_t* __restrict__ subset_sizes,     // Number of blocks per hint
    const uint32_t* __restrict__ extra_blocks,     // Extra block per hint (UINT32_MAX if none)
    const uint32_t* __restrict__ extra_offsets,    // Extra offset per hint
    const uint8_t* __restrict__ entries,
    HintOutput* __restrict__ output,
    HintOutput* __restrict__ backup_high_output    // For backup hints (not yet implemented)
) {
    uint32_t hint_idx = blockIdx.x;
    if (hint_idx >= params.total_hints) return;

    uint32_t tid = threadIdx.x;
    uint32_t num_threads = blockDim.x;  // 256

    // Shared memory for partial XOR results - each warp writes 5 values
    __shared__ uint64_t shared_parity[5 * 32];  // 32 warps max (1280 bytes)

    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;
    uint32_t num_warps = (num_threads + 31) / 32;

    // Local accumulator for this thread
    uint64_t local_parity[5] = {0, 0, 0, 0, 0};

    // Load hint's subset bounds
    uint32_t start = subset_starts[hint_idx];
    uint32_t size = subset_sizes[hint_idx];

    // Each thread processes a strided subset of blocks
    for (uint32_t i = tid; i < size; i += num_threads) {
        uint32_t block_idx = subset_blocks[start + i];
        uint32_t offset = subset_offsets[start + i];

        uint64_t entry_idx = (uint64_t)block_idx * params.block_size + offset;
        if (entry_idx >= params.num_entries) continue;

        const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
        local_parity[0] ^= entry_ptr[0];
        local_parity[1] ^= entry_ptr[1];
        local_parity[2] ^= entry_ptr[2];
        local_parity[3] ^= entry_ptr[3];
        local_parity[4] ^= entry_ptr[4];
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            uint32_t lo = (uint32_t)local_parity[i];
            uint32_t hi = (uint32_t)(local_parity[i] >> 32);
            lo ^= __shfl_xor_sync(0xFFFFFFFF, lo, offset);
            hi ^= __shfl_xor_sync(0xFFFFFFFF, hi, offset);
            local_parity[i] = ((uint64_t)hi << 32) | lo;
        }
    }

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        shared_parity[warp_id * 5 + 0] = local_parity[0];
        shared_parity[warp_id * 5 + 1] = local_parity[1];
        shared_parity[warp_id * 5 + 2] = local_parity[2];
        shared_parity[warp_id * 5 + 3] = local_parity[3];
        shared_parity[warp_id * 5 + 4] = local_parity[4];
    }

    __syncthreads();

    // Final reduction by first warp - each of first 5 threads handles one parity element
    if (warp_id == 0 && lane_id < 5) {
        uint64_t final_val = 0;
        for (uint32_t w = 0; w < num_warps; w++) {
            final_val ^= shared_parity[w * 5 + lane_id];
        }

        // Handle extra entry (only lane 0 for element 0)
        if (lane_id == 0) {
            uint32_t extra_block = extra_blocks[hint_idx];
            if (extra_block != UINT32_MAX) {
                uint32_t extra_offset = extra_offsets[hint_idx];
                uint64_t extra_entry_idx = (uint64_t)extra_block * params.block_size + extra_offset;
                if (extra_entry_idx < params.num_entries) {
                    const uint64_t* entry_ptr = (const uint64_t*)(entries + extra_entry_idx * ENTRY_SIZE);
                    // XOR extra entry into all 5 elements
                    // Lane 0 does element 0, need to broadcast to other lanes
                    shared_parity[0] = entry_ptr[0];
                    shared_parity[1] = entry_ptr[1];
                    shared_parity[2] = entry_ptr[2];
                    shared_parity[3] = entry_ptr[3];
                    shared_parity[4] = entry_ptr[4];
                }
            }
        }
    }

    __syncthreads();

    // Write output - lanes 0-4 write their respective elements
    if (warp_id == 0 && lane_id < 5) {
        uint64_t final_val = 0;
        for (uint32_t w = 0; w < num_warps; w++) {
            final_val ^= shared_parity[w * 5 + lane_id];
        }

        uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
        out_ptr[lane_id] = final_val;
        if (lane_id == 0) {
            out_ptr[5] = 0;  // Padding
        }
    }

    (void)backup_high_output;
}

// ============================================================================
// Forge-Optimized Kernel V2 (Simplified, single-pass reduction)
// ============================================================================

/**
 * Forge-optimized V2: Cleaner reduction logic.
 * 
 * Same performance as V1 but with simpler code path.
 */
extern "C" __global__ void rms24_hint_gen_forge_v2_kernel(
    const Rms24Params params,
    const uint32_t* __restrict__ subset_blocks,
    const uint32_t* __restrict__ subset_offsets,
    const uint32_t* __restrict__ subset_starts,
    const uint32_t* __restrict__ subset_sizes,
    const uint32_t* __restrict__ extra_blocks,
    const uint32_t* __restrict__ extra_offsets,
    const uint8_t* __restrict__ entries,
    HintOutput* __restrict__ output,
    HintOutput* __restrict__ backup_high_output
) {
    uint32_t hint_idx = blockIdx.x;
    if (hint_idx >= params.total_hints) return;

    uint32_t tid = threadIdx.x;
    uint32_t num_threads = blockDim.x;

    __shared__ uint64_t shared_parity[5 * 32];

    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;
    uint32_t num_warps = (num_threads + 31) / 32;

    uint64_t local_parity[5] = {0, 0, 0, 0, 0};

    uint32_t start = subset_starts[hint_idx];
    uint32_t size = subset_sizes[hint_idx];

    // Strided iteration
    for (uint32_t i = tid; i < size; i += num_threads) {
        uint32_t block_idx = subset_blocks[start + i];
        uint32_t offset = subset_offsets[start + i];
        uint64_t entry_idx = (uint64_t)block_idx * params.block_size + offset;

        if (entry_idx < params.num_entries) {
            const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
            local_parity[0] ^= entry_ptr[0];
            local_parity[1] ^= entry_ptr[1];
            local_parity[2] ^= entry_ptr[2];
            local_parity[3] ^= entry_ptr[3];
            local_parity[4] ^= entry_ptr[4];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_parity[0] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[0], offset);
        local_parity[1] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[1], offset);
        local_parity[2] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[2], offset);
        local_parity[3] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[3], offset);
        local_parity[4] ^= __shfl_xor_sync(0xFFFFFFFF, local_parity[4], offset);
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            shared_parity[warp_id * 5 + i] = local_parity[i];
        }
    }

    __syncthreads();

    // Final reduction and output by thread 0
    if (tid == 0) {
        uint64_t final_parity[5] = {0, 0, 0, 0, 0};

        for (uint32_t w = 0; w < num_warps; w++) {
            final_parity[0] ^= shared_parity[w * 5 + 0];
            final_parity[1] ^= shared_parity[w * 5 + 1];
            final_parity[2] ^= shared_parity[w * 5 + 2];
            final_parity[3] ^= shared_parity[w * 5 + 3];
            final_parity[4] ^= shared_parity[w * 5 + 4];
        }

        // Extra entry
        uint32_t extra_block = extra_blocks[hint_idx];
        if (extra_block != UINT32_MAX) {
            uint32_t extra_offset = extra_offsets[hint_idx];
            uint64_t extra_entry_idx = (uint64_t)extra_block * params.block_size + extra_offset;
            if (extra_entry_idx < params.num_entries) {
                const uint64_t* entry_ptr = (const uint64_t*)(entries + extra_entry_idx * ENTRY_SIZE);
                final_parity[0] ^= entry_ptr[0];
                final_parity[1] ^= entry_ptr[1];
                final_parity[2] ^= entry_ptr[2];
                final_parity[3] ^= entry_ptr[3];
                final_parity[4] ^= entry_ptr[4];
            }
        }

        uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
        out_ptr[0] = final_parity[0];
        out_ptr[1] = final_parity[1];
        out_ptr[2] = final_parity[2];
        out_ptr[3] = final_parity[3];
        out_ptr[4] = final_parity[4];
        out_ptr[5] = 0;
    }

    (void)backup_high_output;
}
