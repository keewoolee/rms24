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

    #[test]
    fn test_median_cutoff_collision() {
        // Two middle values are the same
        let values = vec![10, 20, 20, 40];
        let cutoff = find_median_cutoff(&values);
        assert_eq!(cutoff, 0); // Collision returns 0
    }
}
