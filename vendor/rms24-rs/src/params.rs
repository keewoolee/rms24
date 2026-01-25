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
