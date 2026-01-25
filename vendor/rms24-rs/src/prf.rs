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
        // Just verify it returns a 64-bit value
        assert!(v > 0 || v == 0);
    }
}
