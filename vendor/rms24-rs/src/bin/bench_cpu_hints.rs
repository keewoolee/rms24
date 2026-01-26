use clap::Parser;
use memmap2::MmapOptions;
use rayon::prelude::*;
use rms24::{client::Client, params::Params, hints::{xor_bytes_inplace, find_median_cutoff}, prf::Prf};
use std::fs::File;
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Parser)]
struct Args {
    /// Path to database file
    #[arg(long)]
    db: String,

    /// Entry size in bytes
    #[arg(long, default_value = "40")]
    entry_size: usize,

    /// Security parameter (lambda)
    #[arg(long, default_value = "80")]
    lambda: u32,

    /// Number of threads (defaults to rayon auto-detect)
    #[arg(long)]
    threads: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global()?;
    }

    println!("Opening database: {}", args.db);
    let file = File::open(&args.db)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let db_size = mmap.len();
    let num_entries = db_size / args.entry_size;

    println!("Database size: {:.2} GB", db_size as f64 / 1e9);
    println!("Entries: {}", num_entries);
    
    let params = Params::new(num_entries as u64, args.entry_size, args.lambda);
    println!("Params: {:?}", params);

    let num_total_hints = (params.num_reg_hints + params.num_backup_hints) as usize;
    println!("Total Hints: {}", num_total_hints);

    let prf = Prf::random();
    let entry_size = args.entry_size;
    let num_blocks = params.num_blocks as u32;
    let block_size = params.block_size;
    let num_reg = params.num_reg_hints as usize;

    println!("Starting Streamed Hint Generation (Hint-Parallel)...");
    let start = Instant::now();

    let hints_done = AtomicUsize::new(0);

    let mut parities = vec![0u8; num_total_hints * entry_size];
    let mut backup_parities_high = vec![0u8; (num_total_hints - num_reg) * entry_size];

    let parities_ptr = parities.as_mut_ptr() as usize;
    let backup_ptr = backup_parities_high.as_mut_ptr() as usize;

    (0..num_total_hints).into_par_iter()
        .map_init(
            || (
                Vec::with_capacity(num_blocks as usize), 
                Vec::with_capacity(num_blocks as usize),
                Vec::with_capacity(num_blocks as usize * 64),
                Vec::with_capacity(num_blocks as usize * 64)
            ),
            |(select_values, offset_values, select_bytes, offset_bytes), hint_idx| {
                // Phase 1: Generate subset for THIS hint
                prf.fill_select_and_offset_reused(
                    hint_idx as u32, 
                    num_blocks, 
                    select_values, 
                    offset_values,
                    select_bytes,
                    offset_bytes
                );
                let cutoff = find_median_cutoff(select_values);
                
                if cutoff == 0 {
                    return;
                }

                let my_parity = unsafe {
                    std::slice::from_raw_parts_mut((parities_ptr + hint_idx * entry_size) as *mut u8, entry_size)
                };
                
                let is_regular = hint_idx < num_reg;
                
                for block in 0..num_blocks {
                    let select_val = select_values[block as usize];
                    let offset_val = offset_values[block as usize];
                    let picked_offset = offset_val % block_size;
                    let entry_idx = (block as u64 * block_size) + picked_offset;

                    if entry_idx >= params.num_entries {
                        continue;
                    }

                    let file_offset = entry_idx as usize * entry_size;
                    let entry_data = &mmap[file_offset..file_offset + entry_size];

                    if is_regular {
                        if select_val < cutoff {
                            xor_bytes_inplace(my_parity, entry_data);
                        }
                    } else {
                        if select_val < cutoff {
                            xor_bytes_inplace(my_parity, entry_data);
                        } else {
                            let backup_idx = hint_idx - num_reg;
                            let my_backup_parity = unsafe {
                                std::slice::from_raw_parts_mut((backup_ptr + backup_idx * entry_size) as *mut u8, entry_size)
                            };
                            xor_bytes_inplace(my_backup_parity, entry_data);
                        }
                    }
                }

                if is_regular {
                    use rand::{Rng, SeedableRng};
                    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(hint_idx as u64);
                    
                    let mut high_blocks = Vec::new();
                    for block in 0..num_blocks {
                        if select_values[block as usize] >= cutoff {
                            high_blocks.push(block);
                        }
                    }
                    
                    if !high_blocks.is_empty() {
                        let block_idx = high_blocks[rng.gen_range(0..high_blocks.len())];
                        let extra_offset = rng.gen_range(0..block_size as u32);
                        let extra_idx = (block_idx as u64 * block_size) + extra_offset as u64;
                        if extra_idx < params.num_entries {
                            let file_offset = extra_idx as usize * entry_size;
                            let entry_data = &mmap[file_offset..file_offset + entry_size];
                            xor_bytes_inplace(my_parity, entry_data);
                        }
                    }
                }

                let done = hints_done.fetch_add(1, Ordering::Relaxed);
                if (done + 1) % 1000 == 0 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = (done + 1) as f64 / elapsed;
                    println!("Processed {}/{} hints ({:.1}%, {:.1} hints/s, est. remaining: {:.1}m)", 
                        done + 1, num_total_hints, (done + 1) as f64 * 100.0 / num_total_hints as f64,
                        rate, (num_total_hints - (done + 1)) as f64 / rate / 60.0);
                }
            }
        ).for_each(|_| {});

    let duration = start.elapsed();
    println!("Total generation complete in {:.2}s", duration.as_secs_f64());
    
    let nonzero = parities.chunks(entry_size).filter(|p| p.iter().any(|&b| b != 0)).count();
    println!("Non-zero parities: {} / {}", nonzero, num_total_hints);
    
    let effective_gb = (db_size as f64 / 1e9) * (args.lambda as f64 * 0.5);
    println!("Effective Throughput: {:.2} GB/s", effective_gb / duration.as_secs_f64());

    Ok(())
}