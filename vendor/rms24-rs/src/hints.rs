//! Hint storage for RMS24.

/// Precomputed subset for a single hint.
/// Used to pass subset data to GPU without per-block PRF calls.
pub struct HintSubset {
    /// Sorted block indices in the subset (blocks with select < cutoff)
    pub blocks: Vec<u32>,
    /// Precomputed offset for each block (prf.offset % block_size)
    pub offsets: Vec<u32>,
    /// Block index for extra entry (regular hints only, u32::MAX if none)
    pub extra_block: u32,
    /// Offset within extra block
    pub extra_offset: u32,
    /// Whether this is a regular hint (true) or backup hint (false)
    pub is_regular: bool,
}

impl HintSubset {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            offsets: Vec::new(),
            extra_block: u32::MAX,
            extra_offset: 0,
            is_regular: true,
        }
    }
}

impl Default for HintSubset {
    fn default() -> Self {
        Self::new()
    }
}

/// Flattened subset data for GPU transfer.
/// All hints' blocks/offsets concatenated with index arrays.
pub struct SubsetData {
    /// All blocks concatenated: [hint0_blocks..., hint1_blocks..., ...]
    pub blocks: Vec<u32>,
    /// All offsets concatenated (same layout as blocks)
    pub offsets: Vec<u32>,
    /// Start index in blocks/offsets for each hint
    pub starts: Vec<u32>,
    /// Number of blocks for each hint
    pub sizes: Vec<u32>,
    /// Extra block per hint (u32::MAX if none)
    pub extra_blocks: Vec<u32>,
    /// Extra offset per hint
    pub extra_offsets: Vec<u32>,
    /// Whether each hint is regular (true) or backup (false)
    pub is_regular: Vec<bool>,
}

impl SubsetData {
    pub fn from_subsets(subsets: &[HintSubset]) -> Self {
        let total_blocks: usize = subsets.iter().map(|s| s.blocks.len()).sum();
        let mut blocks = Vec::with_capacity(total_blocks);
        let mut offsets = Vec::with_capacity(total_blocks);
        let mut starts = Vec::with_capacity(subsets.len());
        let mut sizes = Vec::with_capacity(subsets.len());
        let mut extra_blocks = Vec::with_capacity(subsets.len());
        let mut extra_offsets = Vec::with_capacity(subsets.len());
        let mut is_regular = Vec::with_capacity(subsets.len());

        let mut current_start: u32 = 0;
        for subset in subsets {
            starts.push(current_start);
            sizes.push(subset.blocks.len() as u32);
            blocks.extend_from_slice(&subset.blocks);
            offsets.extend_from_slice(&subset.offsets);
            extra_blocks.push(subset.extra_block);
            extra_offsets.push(subset.extra_offset);
            is_regular.push(subset.is_regular);
            current_start += subset.blocks.len() as u32;
        }

        Self {
            blocks,
            offsets,
            starts,
            sizes,
            extra_blocks,
            extra_offsets,
            is_regular,
        }
    }
}

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
/// Find median cutoff value in-place.
pub fn find_median_cutoff(v: &mut [u32]) -> u32 {
    debug_assert!(v.len() % 2 == 0, "Length must be even");
    let mid = v.len() / 2;
    
    // Find the element that would be at the mid position in a sorted array
    let (_, mid_val, _) = v.select_nth_unstable(mid);
    let cutoff = *mid_val;
    
    // Find the element that would be at mid-1 position to check for collision
    let (_, max_lower_val, _) = v[..mid].select_nth_unstable(mid - 1);
    let max_lower = *max_lower_val;
    
    if max_lower == cutoff {
        return 0; // Collision at median
    }
    cutoff
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

    #[test]
    fn test_hint_subset_new() {
        let subset = HintSubset::new();
        assert!(subset.blocks.is_empty());
        assert!(subset.offsets.is_empty());
        assert_eq!(subset.extra_block, u32::MAX);
        assert_eq!(subset.extra_offset, 0);
        assert!(subset.is_regular);
    }

    #[test]
    fn test_subset_data_from_subsets() {
        let subsets = vec![
            HintSubset {
                blocks: vec![0, 5, 10],
                offsets: vec![100, 200, 300],
                extra_block: 7,
                extra_offset: 42,
                is_regular: true,
            },
            HintSubset {
                blocks: vec![2, 8],
                offsets: vec![50, 60],
                extra_block: u32::MAX,
                extra_offset: 0,
                is_regular: false,
            },
            HintSubset {
                blocks: vec![1],
                offsets: vec![99],
                extra_block: 3,
                extra_offset: 77,
                is_regular: true,
            },
        ];

        let data = SubsetData::from_subsets(&subsets);

        // Check concatenated blocks/offsets
        assert_eq!(data.blocks, vec![0, 5, 10, 2, 8, 1]);
        assert_eq!(data.offsets, vec![100, 200, 300, 50, 60, 99]);

        // Check starts and sizes
        assert_eq!(data.starts, vec![0, 3, 5]);
        assert_eq!(data.sizes, vec![3, 2, 1]);

        // Check extra entries
        assert_eq!(data.extra_blocks, vec![7, u32::MAX, 3]);
        assert_eq!(data.extra_offsets, vec![42, 0, 77]);

        // Check is_regular flags
        assert_eq!(data.is_regular, vec![true, false, true]);
    }
}
