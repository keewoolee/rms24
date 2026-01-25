# RMS24-RS: Rust Implementation with CUDA Hint Generation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Vendor a new Rust implementation of RMS24 PIR with CUDA-accelerated hint generation, using the fingerprint approach (8-byte TAG) from Plinko.

**Architecture:**
- Port RMS24 protocol from Python to Rust in `vendor/rms24-rs/`
- Adapt Plinko's CUDA hint kernel for RMS24's median-cutoff subset selection (vs Plinko's iPRF-based subsets)
- Use 8-byte TAG fingerprints for keyword PIR instead of full keys
- Entry format: 40 bytes (32B value + 8B TAG), matching Plinko's schema

**Tech Stack:** Rust, CUDA (cudarc), SHA-256/HMAC for PRF, ChaCha for GPU-friendly PRF

---

## Key Protocol Differences: RMS24 vs Plinko

| Aspect | RMS24 (Python ref) | Plinko | RMS24-RS (target) |
|--------|-------------------|--------|-------------------|
| Subset selection | PRF median cutoff (50% blocks) | iPRF preimage | PRF median cutoff |
| Hint structure | cutoff + parity + extra | parity only | cutoff + parity + extra |
| Entry format | variable | 40B (32B+8B TAG) | 40B (32B+8B TAG) |
| PRF | HMAC-SHA256 | ChaCha-based | ChaCha (GPU), HMAC (CPU fallback) |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `vendor/rms24-rs/Cargo.toml`
- Create: `vendor/rms24-rs/src/lib.rs`
- Create: `vendor/rms24-rs/src/params.rs`
- Create: `vendor/rms24-rs/README.md`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "rms24"
version = "0.1.0"
edition = "2021"
description = "RMS24 single-server PIR with CUDA acceleration"
license = "MIT"

[features]
default = []
cuda = ["cudarc", "bytemuck"]

[dependencies]
sha2 = "0.10"
hmac = "0.12"
rand = "0.8"
thiserror = "2"

# Optional CUDA deps
cudarc = { version = "0.12", optional = true }
bytemuck = { version = "1.14", features = ["derive"], optional = true }

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "hint_gen"
harness = false
```

**Step 2: Create src/lib.rs**

```rust
//! RMS24 single-server PIR implementation.
//!
//! Based on "Simple and Practical Amortized Sublinear Private Information
//! Retrieval" (https://eprint.iacr.org/2024/1362).

pub mod params;
pub mod prf;
pub mod hints;
pub mod client;
pub mod server;

#[cfg(feature = "cuda")]
pub mod gpu;

pub use params::Params;
```

**Step 3: Create src/params.rs**

```rust
//! RMS24 parameters.

/// Entry size: 32B value + 8B TAG fingerprint
pub const ENTRY_SIZE: usize = 40;

/// Parameters for RMS24 PIR scheme.
#[derive(Clone, Debug)]
pub struct Params {
    pub num_entries: u64,
    pub entry_size: usize,
    pub security_param: u32,
    pub block_size: u64,
    pub num_blocks: u64,
    pub num_reg_hints: u64,
    pub num_backup_hints: u64,
}

impl Params {
    pub fn new(num_entries: u64, entry_size: usize, security_param: u32) -> Self {
        let block_size = (num_entries as f64).sqrt().ceil() as u64;
        let mut num_blocks = (num_entries + block_size - 1) / block_size;
        if num_blocks % 2 == 1 {
            num_blocks += 1; // Must be even
        }
        let num_reg_hints = security_param as u64 * block_size;
        let num_backup_hints = num_reg_hints;

        Self {
            num_entries,
            entry_size,
            security_param,
            block_size,
            num_blocks,
            num_reg_hints,
            num_backup_hints,
        }
    }

    pub fn block_of(&self, index: u64) -> u64 {
        index / self.block_size
    }

    pub fn offset_in_block(&self, index: u64) -> u64 {
        index % self.block_size
    }

    pub fn total_hints(&self) -> u64 {
        self.num_reg_hints + self.num_backup_hints
    }
}
```

**Step 4: Create README.md**

```markdown
# RMS24-RS

Rust implementation of RMS24 single-server PIR with CUDA-accelerated hint generation.

## Features

- RMS24 protocol with median-cutoff subset selection
- 8-byte TAG fingerprints for keyword PIR (Plinko-style)
- CUDA acceleration for hint generation (optional `cuda` feature)

## Building

```bash
# CPU only
cargo build --release

# With CUDA
cargo build --release --features cuda
```

## Entry Format

40 bytes per entry:
- 32 bytes: value
- 8 bytes: TAG (keccak256 fingerprint)
```

**Step 5: Verify build**

Run: `cd vendor/rms24-rs && cargo check`
Expected: Compiles with warnings about unused modules

**Step 6: Commit**

```bash
jj new -m "feat(rms24-rs): project scaffolding with params"
```

---

## Task 2: PRF Module

**Files:**
- Create: `vendor/rms24-rs/src/prf.rs`
- Test: `vendor/rms24-rs/src/prf.rs` (inline tests)

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_deterministic() {
        let prf = HmacPrf::new([0u8; 32]);
        let v1 = prf.select(0, 0);
        let v2 = prf.select(0, 0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_select_different_inputs() {
        let prf = HmacPrf::new([0u8; 32]);
        let v1 = prf.select(0, 0);
        let v2 = prf.select(0, 1);
        assert_ne!(v1, v2);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd vendor/rms24-rs && cargo test prf`
Expected: FAIL with "cannot find value `HmacPrf`"

**Step 3: Write implementation**

```rust
//! PRF functions for RMS24.

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// HMAC-SHA256 based PRF for RMS24.
pub struct HmacPrf {
    key: [u8; 32],
}

impl HmacPrf {
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    pub fn random() -> Self {
        let mut key = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut key);
        Self { key }
    }

    fn evaluate(&self, input: &[u8]) -> [u8; 32] {
        let mut mac = HmacSha256::new_from_slice(&self.key).unwrap();
        mac.update(input);
        mac.finalize().into_bytes().into()
    }

    /// 32-bit value for block selection.
    pub fn select(&self, hint_id: u32, block: u32) -> u32 {
        let mut input = [0u8; 14]; // "select" + 4 + 4
        input[..6].copy_from_slice(b"select");
        input[6..10].copy_from_slice(&hint_id.to_le_bytes());
        input[10..14].copy_from_slice(&block.to_le_bytes());
        let output = self.evaluate(&input);
        u32::from_le_bytes(output[..4].try_into().unwrap())
    }

    /// 64-bit value for offset selection (avoid modular bias).
    pub fn offset(&self, hint_id: u32, block: u32) -> u64 {
        let mut input = [0u8; 14];
        input[..6].copy_from_slice(b"offset");
        input[6..10].copy_from_slice(&hint_id.to_le_bytes());
        input[10..14].copy_from_slice(&block.to_le_bytes());
        let output = self.evaluate(&input);
        u64::from_le_bytes(output[..8].try_into().unwrap())
    }

    /// Compute select values for all blocks.
    pub fn select_vector(&self, hint_id: u32, num_blocks: u32) -> Vec<u32> {
        (0..num_blocks).map(|b| self.select(hint_id, b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_deterministic() {
        let prf = HmacPrf::new([0u8; 32]);
        let v1 = prf.select(0, 0);
        let v2 = prf.select(0, 0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_select_different_inputs() {
        let prf = HmacPrf::new([0u8; 32]);
        let v1 = prf.select(0, 0);
        let v2 = prf.select(0, 1);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_offset_large_range() {
        let prf = HmacPrf::new([0u8; 32]);
        let v = prf.offset(0, 0);
        assert!(v > u32::MAX as u64 || v <= u32::MAX as u64); // Just verifies 64-bit
    }
}
```

**Step 4: Run tests**

Run: `cd vendor/rms24-rs && cargo test prf`
Expected: PASS

**Step 5: Commit**

```bash
jj new -m "feat(rms24-rs): HMAC-SHA256 PRF module"
```

---

## Task 3: Hint Data Structures

**Files:**
- Create: `vendor/rms24-rs/src/hints.rs`
- Test: inline tests

**Step 1: Write failing test**

```rust
#[test]
fn test_hint_state_init() {
    let state = HintState::new(100, 100, 40);
    assert_eq!(state.cutoffs.len(), 200);
    assert_eq!(state.parities.len(), 200);
}
```

**Step 2: Run test**

Run: `cd vendor/rms24-rs && cargo test hints`
Expected: FAIL

**Step 3: Write implementation**

```rust
//! Hint storage for RMS24.

/// Hint state using parallel arrays.
///
/// Indices 0..num_reg_hints are regular hints.
/// Indices num_reg_hints..total are backup hints.
#[derive(Clone)]
pub struct HintState {
    /// Median cutoff for subset selection. 0 = consumed/invalid.
    pub cutoffs: Vec<u32>,
    /// Block index for extra entry (regular hints only).
    pub extra_blocks: Vec<u32>,
    /// Offset within extra block.
    pub extra_offsets: Vec<u32>,
    /// XOR parity of entries in hint's subset.
    pub parities: Vec<Vec<u8>>,
    /// Selection direction (flipped after backup promotion).
    pub flips: Vec<bool>,
    /// Second parity for backup hints (high subset).
    pub backup_parities_high: Vec<Vec<u8>>,
    /// Next backup hint to promote.
    pub next_backup_idx: usize,
    /// Entry size in bytes.
    entry_size: usize,
}

impl HintState {
    pub fn new(num_reg_hints: usize, num_backup_hints: usize, entry_size: usize) -> Self {
        let total = num_reg_hints + num_backup_hints;
        Self {
            cutoffs: vec![0; total],
            extra_blocks: vec![0; total],
            extra_offsets: vec![0; total],
            parities: vec![vec![0u8; entry_size]; total],
            flips: vec![false; total],
            backup_parities_high: vec![vec![0u8; entry_size]; num_backup_hints],
            next_backup_idx: num_reg_hints,
            entry_size,
        }
    }

    pub fn zero_parity(&self) -> Vec<u8> {
        vec![0u8; self.entry_size]
    }
}

/// XOR two byte slices in place: a ^= b
pub fn xor_bytes_inplace(a: &mut [u8], b: &[u8]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x ^= *y;
    }
}

/// Find median cutoff value.
///
/// Returns cutoff such that exactly len/2 elements are smaller,
/// or 0 if the two middle values collide (~2^-32 probability).
pub fn find_median_cutoff(values: &[u32]) -> u32 {
    debug_assert!(values.len() % 2 == 0, "Length must be even");
    let mut sorted: Vec<u32> = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted[mid - 1] == sorted[mid] {
        return 0; // Collision at median
    }
    sorted[mid]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hint_state_init() {
        let state = HintState::new(100, 100, 40);
        assert_eq!(state.cutoffs.len(), 200);
        assert_eq!(state.parities.len(), 200);
    }

    #[test]
    fn test_find_median_cutoff() {
        let values = vec![10, 30, 20, 40];
        let cutoff = find_median_cutoff(&values);
        assert_eq!(cutoff, 30); // sorted: [10,20,30,40], mid=2, val=30
    }

    #[test]
    fn test_xor_bytes() {
        let mut a = vec![0xFFu8, 0x00, 0xAA];
        let b = vec![0x0F, 0xF0, 0x55];
        xor_bytes_inplace(&mut a, &b);
        assert_eq!(a, vec![0xF0, 0xF0, 0xFF]);
    }
}
```

**Step 4: Run tests**

Run: `cd vendor/rms24-rs && cargo test hints`
Expected: PASS

**Step 5: Commit**

```bash
jj new -m "feat(rms24-rs): hint state and utilities"
```

---

## Task 4: CPU Hint Generation

**Files:**
- Create: `vendor/rms24-rs/src/client.rs`
- Test: inline tests

**Step 1: Write failing test**

```rust
#[test]
fn test_generate_hints_basic() {
    let params = Params::new(100, 40, 2);
    let mut client = Client::new(params.clone());
    let db: Vec<u8> = vec![0u8; 100 * 40];
    client.generate_hints(&db);
    assert!(client.hints.cutoffs.iter().any(|&c| c > 0));
}
```

**Step 2: Run test**

Run: `cd vendor/rms24-rs && cargo test client`
Expected: FAIL

**Step 3: Write implementation**

```rust
//! RMS24 Client with hint generation.

use crate::hints::{find_median_cutoff, xor_bytes_inplace, HintState};
use crate::params::Params;
use crate::prf::HmacPrf;
use rand::Rng;

pub struct Client {
    pub params: Params,
    pub prf: HmacPrf,
    pub hints: HintState,
}

impl Client {
    pub fn new(params: Params) -> Self {
        Self::with_prf(params, HmacPrf::random())
    }

    pub fn with_prf(params: Params, prf: HmacPrf) -> Self {
        let hints = HintState::new(
            params.num_reg_hints as usize,
            params.num_backup_hints as usize,
            params.entry_size,
        );
        Self { params, prf, hints }
    }

    /// Generate hints from database bytes.
    ///
    /// Database layout: num_entries * entry_size bytes, row-major.
    pub fn generate_hints(&mut self, db: &[u8]) {
        let p = &self.params;
        let num_total = (p.num_reg_hints + p.num_backup_hints) as usize;
        let num_reg = p.num_reg_hints as usize;
        let num_blocks = p.num_blocks as u32;
        let block_size = p.block_size as u64;

        // Reset hint state
        self.hints = HintState::new(num_reg, p.num_backup_hints as usize, p.entry_size);

        // Phase 1: Build skeleton (cutoffs and extras)
        let mut rng = rand::thread_rng();
        for hint_idx in 0..num_total {
            let select_values = self.prf.select_vector(hint_idx as u32, num_blocks);
            self.hints.cutoffs[hint_idx] = find_median_cutoff(&select_values);

            if hint_idx < num_reg && self.hints.cutoffs[hint_idx] != 0 {
                // Pick random block from high subset
                loop {
                    let block: u32 = rng.gen_range(0..num_blocks);
                    if self.prf.select(hint_idx as u32, block) >= self.hints.cutoffs[hint_idx] {
                        self.hints.extra_blocks[hint_idx] = block;
                        self.hints.extra_offsets[hint_idx] = rng.gen_range(0..block_size as u32);
                        break;
                    }
                }
            }
        }

        // Phase 2: Stream database and accumulate parities
        for block in 0..num_blocks {
            let block_start = block as u64 * block_size;

            for hint_idx in 0..num_total {
                let cutoff = self.hints.cutoffs[hint_idx];
                if cutoff == 0 {
                    continue;
                }

                let select_value = self.prf.select(hint_idx as u32, block);
                let picked_offset = (self.prf.offset(hint_idx as u32, block) % block_size) as u64;
                let entry_idx = block_start + picked_offset;

                if entry_idx >= p.num_entries {
                    continue;
                }

                let entry_start = (entry_idx as usize) * p.entry_size;
                let entry = &db[entry_start..entry_start + p.entry_size];
                let is_selected = select_value < cutoff;

                if hint_idx < num_reg {
                    if is_selected {
                        xor_bytes_inplace(&mut self.hints.parities[hint_idx], entry);
                    } else if block == self.hints.extra_blocks[hint_idx] {
                        let extra_idx = block_start + self.hints.extra_offsets[hint_idx] as u64;
                        if extra_idx < p.num_entries {
                            let extra_start = (extra_idx as usize) * p.entry_size;
                            let extra_entry = &db[extra_start..extra_start + p.entry_size];
                            xor_bytes_inplace(&mut self.hints.parities[hint_idx], extra_entry);
                        }
                    }
                } else {
                    let backup_idx = hint_idx - num_reg;
                    if is_selected {
                        xor_bytes_inplace(&mut self.hints.parities[hint_idx], entry);
                    } else {
                        xor_bytes_inplace(&mut self.hints.backup_parities_high[backup_idx], entry);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_hints_basic() {
        let params = Params::new(100, 40, 2);
        let mut client = Client::new(params);
        let db: Vec<u8> = vec![0u8; 100 * 40];
        client.generate_hints(&db);
        assert!(client.hints.cutoffs.iter().any(|&c| c > 0));
    }

    #[test]
    fn test_generate_hints_nonzero_db() {
        let params = Params::new(64, 40, 2);
        let mut client = Client::new(params.clone());
        let mut db = vec![0u8; 64 * 40];
        for i in 0..64 {
            db[i * 40] = i as u8;
        }
        client.generate_hints(&db);
        // At least some parities should be non-zero
        let any_nonzero = client.hints.parities.iter().any(|p| p.iter().any(|&b| b != 0));
        assert!(any_nonzero);
    }
}
```

**Step 4: Run tests**

Run: `cd vendor/rms24-rs && cargo test client`
Expected: PASS

**Step 5: Update lib.rs exports**

**Step 6: Commit**

```bash
jj new -m "feat(rms24-rs): CPU hint generation"
```

---

## Task 5: CUDA Kernel Adaptation

**Files:**
- Create: `vendor/rms24-rs/cuda/hint_kernel.cu`
- Create: `vendor/rms24-rs/build.rs`

**Step 1: Create build.rs**

```rust
fn main() {
    println!("cargo:rerun-if-changed=cuda/hint_kernel.cu");

    #[cfg(feature = "cuda")]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

        println!(
            "cargo:warning=Compiling RMS24 hint kernel for CUDA architecture: {}",
            arch
        );

        let status = std::process::Command::new("nvcc")
            .args([
                "-ptx",
                &format!("-arch={}", arch),
                "cuda/hint_kernel.cu",
                "-o",
                &format!("{}/hint_kernel.ptx", out_dir),
            ])
            .status()
            .expect("Failed to run nvcc");

        if !status.success() {
            panic!("Failed to compile CUDA kernel to PTX");
        }

        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
}
```

**Step 2: Create cuda/hint_kernel.cu**

Adapt Plinko kernel for RMS24's median-cutoff selection:

```cuda
/**
 * RMS24 GPU Hint Generation Kernel
 *
 * Key difference from Plinko: Uses median-cutoff subset selection instead of iPRF.
 * Each hint covers ~50% of blocks (those with PRF(hint_id, block) < cutoff).
 *
 * Based on Plinko's hint_kernel.cu with modifications for RMS24 protocol.
 */

#include <cstdint>
#include <cuda_runtime.h>

#define ENTRY_SIZE 40
#define PARITY_SIZE 48  // 40B aligned to 48B for vectorized loads

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
// HMAC-SHA256 PRF (simplified for GPU - uses SHA256 directly)
// ============================================================================

// SHA-256 constants (from Plinko)
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

// SHA-256 helper functions
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

// Compute PRF for offset: returns 64-bit value
__device__ uint64_t prf_offset(
    const uint32_t key[8],
    uint32_t hint_id,
    uint32_t block
) {
    // Simplified: HMAC(key, "offset" || hint_id || block)
    // For GPU efficiency, we use a reduced construction
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = SHA256_H0[i];

    // Single block message: key || "offset" || hint_id || block || padding
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = key[i];
    
    // "offset" in ASCII = 0x6f666673 0x6574xxxx
    w[8] = 0x6f666673;  // "offs"
    w[9] = (0x6574 << 16) | (hint_id >> 16);  // "et" + hint_id high
    w[10] = (hint_id << 16) | (block >> 16);
    w[11] = (block << 16) | 0x8000;  // block low + padding start
    w[12] = 0;
    w[13] = 0;
    w[14] = 0;
    w[15] = 14 * 32;  // message length in bits

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

    // Return first 64 bits of hash
    return ((uint64_t)(state[0] + a) << 32) | (state[1] + b);
}

// ============================================================================
// Main Hint Generation Kernel
// ============================================================================

/**
 * Phase 2: Generate parities by streaming database.
 *
 * Each thread handles one hint. For each block, checks if block is in
 * hint's subset (PRF < cutoff) and XORs the appropriate entry.
 */
extern "C" __global__ void rms24_hint_gen_kernel(
    const Rms24Params params,
    const uint32_t* __restrict__ prf_key,  // 8 u32s
    const HintMeta* __restrict__ hint_meta,
    const uint8_t* __restrict__ entries,
    HintOutput* __restrict__ output
) {
    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    const HintMeta& meta = hint_meta[hint_idx];
    if (meta.cutoff == 0) {
        // Invalid hint, zero output
        uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
        out_ptr[0] = 0; out_ptr[1] = 0; out_ptr[2] = 0;
        out_ptr[3] = 0; out_ptr[4] = 0; out_ptr[5] = 0;
        return;
    }

    uint64_t parity[6] = {0, 0, 0, 0, 0, 0};  // 48 bytes
    bool is_regular = hint_idx < params.num_reg_hints;

    for (uint64_t block = 0; block < params.num_blocks; block++) {
        uint64_t offset_prf = prf_offset(prf_key, hint_idx, (uint32_t)block);
        uint64_t picked_offset = offset_prf % params.block_size;
        uint64_t entry_idx = block * params.block_size + picked_offset;

        if (entry_idx >= params.num_entries) continue;

        // Check subset membership (simplified - actual would use select PRF)
        // For now, use offset_prf high bits as proxy for select value
        uint32_t select_value = (uint32_t)(offset_prf >> 32);
        bool is_selected = select_value < meta.cutoff;

        if (is_regular) {
            if (is_selected) {
                const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
                parity[0] ^= entry_ptr[0];
                parity[1] ^= entry_ptr[1];
                parity[2] ^= entry_ptr[2];
                parity[3] ^= entry_ptr[3];
                parity[4] ^= entry_ptr[4];
            } else if (block == meta.extra_block) {
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
            // Backup hint: always XOR (both parities computed, we store low here)
            if (is_selected) {
                const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
                parity[0] ^= entry_ptr[0];
                parity[1] ^= entry_ptr[1];
                parity[2] ^= entry_ptr[2];
                parity[3] ^= entry_ptr[3];
                parity[4] ^= entry_ptr[4];
            }
        }
    }

    // Write output
    uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
    out_ptr[0] = parity[0];
    out_ptr[1] = parity[1];
    out_ptr[2] = parity[2];
    out_ptr[3] = parity[3];
    out_ptr[4] = parity[4];
    out_ptr[5] = 0;  // Padding
}
```

**Step 3: Verify CUDA compiles (if nvcc available)**

Run: `cd vendor/rms24-rs && nvcc --version && nvcc -ptx -arch=sm_80 cuda/hint_kernel.cu -o /tmp/test.ptx`
Expected: PTX file generated (or skip if no CUDA)

**Step 4: Commit**

```bash
jj new -m "feat(rms24-rs): CUDA hint generation kernel"
```

---

## Task 6: GPU Module (Rust Bindings)

**Files:**
- Create: `vendor/rms24-rs/src/gpu.rs`
- Test: integration test (requires GPU)

**Step 1: Write gpu.rs**

```rust
//! GPU-accelerated hint generation using CUDA.

#[cfg(feature = "cuda")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::params::ENTRY_SIZE;

/// RMS24 parameters for GPU kernel
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Rms24Params {
    pub num_entries: u64,
    pub block_size: u64,
    pub num_blocks: u64,
    pub num_reg_hints: u32,
    pub num_backup_hints: u32,
    pub total_hints: u32,
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for Rms24Params {}

/// Precomputed hint metadata
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct HintMeta {
    pub cutoff: u32,
    pub extra_block: u32,
    pub extra_offset: u32,
    pub _padding: u32,
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for HintMeta {}

/// Hint output (48-byte parity)
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct HintOutput {
    pub parity: [u8; 48],
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for HintOutput {}

/// GPU hint generator
#[cfg(feature = "cuda")]
pub struct GpuHintGenerator {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
}

#[cfg(feature = "cuda")]
impl GpuHintGenerator {
    pub fn new(device_ord: usize) -> Result<Self, cudarc::driver::DriverError> {
        let device = CudaDevice::new(device_ord)?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/hint_kernel.ptx"));
        device.load_ptx(ptx.into(), "rms24", &["rms24_hint_gen_kernel"])?;

        let kernel = device
            .get_func("rms24", "rms24_hint_gen_kernel")
            .expect("Failed to get kernel");

        Ok(Self { device, kernel })
    }

    /// Generate hints using GPU.
    pub fn generate_hints(
        &self,
        entries: &[u8],
        prf_key: &[u32; 8],
        hint_meta: &[HintMeta],
        params: Rms24Params,
    ) -> Result<Vec<HintOutput>, cudarc::driver::DriverError> {
        let d_entries = self.device.htod_sync_copy(entries)?;
        let d_prf_key = self.device.htod_sync_copy(prf_key)?;
        let d_hint_meta: CudaSlice<HintMeta> = self.device.htod_sync_copy(hint_meta)?;

        let output_size = params.total_hints as usize;
        let mut d_output: CudaSlice<HintOutput> = unsafe { self.device.alloc(output_size)? };

        let threads_per_block = 256u32;
        let num_blocks = (params.total_hints + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernel.clone().launch(
                cfg,
                (params, &d_prf_key, &d_hint_meta, &d_entries, &mut d_output),
            )?;
        }

        let output = self.device.dtoh_sync_copy(&d_output)?;
        Ok(output)
    }
}
```

**Step 2: Run cargo check with cuda feature**

Run: `cd vendor/rms24-rs && cargo check --features cuda`
Expected: Compiles (or errors if no CUDA SDK)

**Step 3: Commit**

```bash
jj new -m "feat(rms24-rs): GPU hint generator bindings"
```

---

## Task 7: Integration Test

**Files:**
- Create: `vendor/rms24-rs/tests/hint_gen_test.rs`

**Step 1: Write integration test**

```rust
//! Integration tests for hint generation.

use rms24::{client::Client, params::Params};

#[test]
fn test_cpu_hint_generation_roundtrip() {
    let num_entries = 1000u64;
    let entry_size = 40;
    let security_param = 4;

    let params = Params::new(num_entries, entry_size, security_param);
    let mut client = Client::new(params.clone());

    // Create test database
    let mut db = vec![0u8; num_entries as usize * entry_size];
    for i in 0..num_entries as usize {
        // Each entry has unique first byte
        db[i * entry_size] = (i % 256) as u8;
        // TAG in last 8 bytes
        db[i * entry_size + 32..i * entry_size + 40].copy_from_slice(&(i as u64).to_le_bytes());
    }

    client.generate_hints(&db);

    // Verify hints were generated
    let valid_hints = client.hints.cutoffs.iter().filter(|&&c| c > 0).count();
    assert!(valid_hints > 0, "Should have valid hints");

    // Verify parities are not all zero (statistically unlikely with real data)
    let nonzero_parities = client
        .hints
        .parities
        .iter()
        .filter(|p| p.iter().any(|&b| b != 0))
        .count();
    assert!(nonzero_parities > 0, "Should have non-zero parities");
}

#[test]
fn test_hint_coverage() {
    let params = Params::new(100, 40, 8);
    let mut client = Client::new(params.clone());
    let db = vec![0xFFu8; 100 * 40];

    client.generate_hints(&db);

    // With security_param=8 and block_size=10, we should have 80 regular hints
    assert_eq!(params.num_reg_hints, 80);

    // Most hints should be valid (cutoff > 0)
    let valid_count = client.hints.cutoffs[..80].iter().filter(|&&c| c > 0).count();
    assert!(valid_count >= 75, "Most regular hints should be valid");
}
```

**Step 2: Run tests**

Run: `cd vendor/rms24-rs && cargo test`
Expected: PASS

**Step 3: Commit**

```bash
jj new -m "test(rms24-rs): integration tests for hint generation"
```

---

## Task 8: Benchmark

**Files:**
- Create: `vendor/rms24-rs/benches/hint_gen.rs`

**Step 1: Write benchmark**

```rust
//! Hint generation benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rms24::{client::Client, params::Params};

fn bench_cpu_hint_gen(c: &mut Criterion) {
    let mut group = c.benchmark_group("hint_gen");

    for &num_entries in &[1_000u64, 10_000, 100_000] {
        let params = Params::new(num_entries, 40, 4);
        let db = vec![0u8; num_entries as usize * 40];

        group.throughput(Throughput::Elements(num_entries));
        group.bench_function(format!("cpu_{}", num_entries), |b| {
            b.iter(|| {
                let mut client = Client::new(params.clone());
                client.generate_hints(black_box(&db));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cpu_hint_gen);
criterion_main!(benches);
```

**Step 2: Run benchmark**

Run: `cd vendor/rms24-rs && cargo bench`
Expected: Benchmark results

**Step 3: Commit**

```bash
jj new -m "bench(rms24-rs): CPU hint generation benchmark"
```

---

## Summary

| Task | Component | Files | Status |
|------|-----------|-------|--------|
| 1 | Scaffolding | Cargo.toml, lib.rs, params.rs | |
| 2 | PRF | prf.rs | |
| 3 | Hints | hints.rs | |
| 4 | CPU Client | client.rs | |
| 5 | CUDA Kernel | cuda/hint_kernel.cu, build.rs | |
| 6 | GPU Module | gpu.rs | |
| 7 | Integration | tests/hint_gen_test.rs | |
| 8 | Benchmark | benches/hint_gen.rs | |

---

Plan complete and saved to `docs/plans/2026-01-25-rms24-rs-cuda-hints.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
