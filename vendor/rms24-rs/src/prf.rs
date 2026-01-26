//! PRF functions for RMS24 using ChaCha12.
//!
//! ChaCha12 is used instead of HMAC-SHA256 for performance:
//! - ARX operations only (no memory lookups)
//! - GPU-friendly (same implementation in CUDA kernel)
//! - 12 rounds provides sufficient security margin

use chacha20::cipher::{KeyIvInit, StreamCipher};
use chacha20::ChaCha12;

/// ChaCha12-based PRF for RMS24.
///
/// Uses ChaCha12 in counter mode with domain-separated nonces
/// for select and offset operations.
pub struct Prf {
    key: [u8; 32],
}

impl Prf {
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    pub fn random() -> Self {
        let mut key = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut key);
        Self { key }
    }

    /// Get the raw key bytes (for GPU).
    pub fn key(&self) -> &[u8; 32] {
        &self.key
    }

    /// Get key as 8 x u32 (for GPU kernel).
    pub fn key_u32(&self) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = u32::from_le_bytes(self.key[i * 4..(i + 1) * 4].try_into().unwrap());
        }
        result
    }

    /// Evaluate ChaCha12 PRF with domain separation.
    ///
    /// Nonce layout (12 bytes):
    /// - bytes 0-3: domain tag (0 = select, 1 = offset)
    /// - bytes 4-7: hint_id
    /// - bytes 8-11: block
    fn evaluate(&self, domain: u32, hint_id: u32, block: u32) -> [u8; 64] {
        let mut nonce = [0u8; 12];
        nonce[0..4].copy_from_slice(&domain.to_le_bytes());
        nonce[4..8].copy_from_slice(&hint_id.to_le_bytes());
        nonce[8..12].copy_from_slice(&block.to_le_bytes());

        let mut cipher = ChaCha12::new((&self.key).into(), (&nonce).into());
        let mut output = [0u8; 64];
        cipher.apply_keystream(&mut output);
        output
    }

    /// 32-bit value for block selection.
    ///
    /// Used to determine if a block is in the "low" or "high" subset
    /// relative to the median cutoff.
    pub fn select(&self, hint_id: u32, block: u32) -> u32 {
        let output = self.evaluate(0, hint_id, block);
        u32::from_le_bytes(output[0..4].try_into().unwrap())
    }

    /// 64-bit value for offset selection.
    ///
    /// Uses 64 bits to avoid modular bias when computing offset % block_size.
    pub fn offset(&self, hint_id: u32, block: u32) -> u64 {
        let output = self.evaluate(1, hint_id, block);
        u64::from_le_bytes(output[0..8].try_into().unwrap())
    }

    /// Compute select and offset values for all blocks into provided vectors, reusing byte buffers.
    pub fn fill_select_and_offset_reused(
        &self, 
        hint_id: u32, 
        num_blocks: u32, 
        selects: &mut Vec<u32>, 
        offsets: &mut Vec<u64>,
        select_bytes: &mut Vec<u8>,
        offset_bytes: &mut Vec<u8>,
    ) {
        selects.resize(num_blocks as usize, 0);
        offsets.resize(num_blocks as usize, 0);
        select_bytes.resize(num_blocks as usize * 64, 0);
        offset_bytes.resize(num_blocks as usize * 64, 0);
        
        // Zero out bytes to ensure apply_keystream works correctly if it only XORs
        // Actually chacha20 crate's apply_keystream XORs into the buffer.
        // So we MUST zero it.
        select_bytes.fill(0);
        offset_bytes.fill(0);

        let mut select_nonce = [0u8; 12];
        select_nonce[0..4].copy_from_slice(&0u32.to_le_bytes());
        select_nonce[4..8].copy_from_slice(&hint_id.to_le_bytes());

        let mut offset_nonce = [0u8; 12];
        offset_nonce[0..4].copy_from_slice(&1u32.to_le_bytes());
        offset_nonce[4..8].copy_from_slice(&hint_id.to_le_bytes());

        let mut select_cipher = ChaCha12::new((&self.key).into(), (&select_nonce).into());
        let mut offset_cipher = ChaCha12::new((&self.key).into(), (&offset_nonce).into());

        select_cipher.apply_keystream(select_bytes);
        offset_cipher.apply_keystream(offset_bytes);

        for i in 0..num_blocks as usize {
            selects[i] = u32::from_le_bytes(select_bytes[i*64..i*64+4].try_into().unwrap());
            offsets[i] = u64::from_le_bytes(offset_bytes[i*64..i*64+8].try_into().unwrap());
        }
    }

    /// Compute select and offset values for all blocks into provided vectors.
    pub fn fill_select_and_offset(&self, hint_id: u32, num_blocks: u32, selects: &mut Vec<u32>, offsets: &mut Vec<u64>) {
        selects.resize(num_blocks as usize, 0);
        offsets.resize(num_blocks as usize, 0);
        
        let mut select_nonce = [0u8; 12];
        select_nonce[0..4].copy_from_slice(&0u32.to_le_bytes());
        select_nonce[4..8].copy_from_slice(&hint_id.to_le_bytes());

        let mut offset_nonce = [0u8; 12];
        offset_nonce[0..4].copy_from_slice(&1u32.to_le_bytes());
        offset_nonce[4..8].copy_from_slice(&hint_id.to_le_bytes());

        let mut select_cipher = ChaCha12::new((&self.key).into(), (&select_nonce).into());
        let mut offset_cipher = ChaCha12::new((&self.key).into(), (&offset_nonce).into());

        // We can't directly apply keystream to Vec<u32> because of endianness and alignment safely without unsafe
        // or a temporary byte buffer. A temporary byte buffer is fine.
        let mut select_bytes = vec![0u8; num_blocks as usize * 64];
        let mut offset_bytes = vec![0u8; num_blocks as usize * 64];

        select_cipher.apply_keystream(&mut select_bytes);
        offset_cipher.apply_keystream(&mut offset_bytes);

        for i in 0..num_blocks as usize {
            selects[i] = u32::from_le_bytes(select_bytes[i*64..i*64+4].try_into().unwrap());
            offsets[i] = u64::from_le_bytes(offset_bytes[i*64..i*64+8].try_into().unwrap());
        }
    }

    /// Compute select and offset values for all blocks.
    pub fn select_and_offset_vectors(&self, hint_id: u32, num_blocks: u32) -> (Vec<u32>, Vec<u64>) {
        let mut select_nonce = [0u8; 12];
        select_nonce[0..4].copy_from_slice(&0u32.to_le_bytes());
        select_nonce[4..8].copy_from_slice(&hint_id.to_le_bytes());

        let mut offset_nonce = [0u8; 12];
        offset_nonce[0..4].copy_from_slice(&1u32.to_le_bytes());
        offset_nonce[4..8].copy_from_slice(&hint_id.to_le_bytes());

        let mut select_cipher = ChaCha12::new((&self.key).into(), (&select_nonce).into());
        let mut offset_cipher = ChaCha12::new((&self.key).into(), (&offset_nonce).into());

        let mut selects = Vec::with_capacity(num_blocks as usize);
        let mut offsets = Vec::with_capacity(num_blocks as usize);
        
        let mut buffer = [0u8; 64];
        for _ in 0..num_blocks {
            // Select
            buffer.fill(0);
            select_cipher.apply_keystream(&mut buffer);
            selects.push(u32::from_le_bytes(buffer[0..4].try_into().unwrap()));

            // Offset
            buffer.fill(0);
            offset_cipher.apply_keystream(&mut buffer);
            offsets.push(u64::from_le_bytes(buffer[0..8].try_into().unwrap()));
        }

        (selects, offsets)
    }

    /// Compute select values for all blocks.
    ///
    /// Used in Phase 1 to find the median cutoff.
    pub fn select_vector(&self, hint_id: u32, num_blocks: u32) -> Vec<u32> {
        let mut nonce = [0u8; 12];
        nonce[0..4].copy_from_slice(&0u32.to_le_bytes()); // domain: select
        nonce[4..8].copy_from_slice(&hint_id.to_le_bytes());
        // nonce[8..12] remains 0, we use counter for block index

        let mut cipher = ChaCha12::new((&self.key).into(), (&nonce).into());
        let mut result = Vec::with_capacity(num_blocks as usize);
        
        // Each 64-byte ChaCha block corresponds to one increment of the counter.
        // We take the first 4 bytes of each block.
        let mut buffer = [0u8; 64];
        for _ in 0..num_blocks {
            buffer.fill(0);
            cipher.apply_keystream(&mut buffer);
            result.push(u32::from_le_bytes(buffer[0..4].try_into().unwrap()));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_deterministic() {
        let prf = Prf::new([0u8; 32]);
        let v1 = prf.select(0, 0);
        let v2 = prf.select(0, 0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_select_different_inputs() {
        let prf = Prf::new([0u8; 32]);
        let v1 = prf.select(0, 0);
        let v2 = prf.select(0, 1);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_select_vs_offset_domain_separation() {
        let prf = Prf::new([0u8; 32]);
        let select_val = prf.select(0, 0);
        let offset_val = prf.offset(0, 0) as u32;
        // Different domains should produce different values
        assert_ne!(select_val, offset_val);
    }

    #[test]
    fn test_key_u32_roundtrip() {
        let key = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
        ];
        let prf = Prf::new(key);
        let key_u32 = prf.key_u32();
        
        // Verify first word: 0x04030201 (little-endian)
        assert_eq!(key_u32[0], 0x04030201);
        assert_eq!(key_u32[7], 0x201f1e1d);
    }

    #[test]
    fn test_select_vector() {
        let prf = Prf::new([42u8; 32]);
        let values = prf.select_vector(0, 100);
        assert_eq!(values.len(), 100);
        // All values should be different (statistically)
        let unique: std::collections::HashSet<_> = values.iter().collect();
        assert!(unique.len() > 90); // Allow some collisions
    }
}
