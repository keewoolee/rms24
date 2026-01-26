//! GPU Hint Generation Benchmark for RMS24
//!
//! Usage:
//!   cargo run --release --features cuda --bin bench_gpu_hints -- \
//!     --db data/database.bin --lambda 80 --iterations 5

use clap::{Parser, ValueEnum};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Kernel {
    Old,
    Warp,
}

#[derive(Parser)]
#[command(name = "bench_gpu_hints")]
#[command(about = "Benchmark GPU hint generation for RMS24")]
struct Args {
    /// Path to database file (use "synthetic:SIZE_MB" for generated data)
    #[arg(long)]
    db: String,

    /// Security parameter (lambda)
    #[arg(long, default_value = "80")]
    lambda: u32,

    /// Number of benchmark iterations
    #[arg(long, default_value = "5")]
    iterations: u32,

    /// Number of warmup iterations
    #[arg(long, default_value = "1")]
    warmup: u32,

    /// Maximum hints to generate (for memory-limited benchmarks)
    #[arg(long)]
    max_hints: Option<u32>,

    /// Starting hint index (for distributed generation across GPUs)
    #[arg(long, default_value = "0")]
    hint_start: u32,

    /// Entry size in bytes
    #[arg(long, default_value = "40")]
    entry_size: usize,

    /// GPU kernel to use
    #[arg(long, value_enum, default_value = "warp")]
    kernel: Kernel,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("RMS24 GPU Hint Generation Benchmark");
    println!("====================================");

    let kernel_name = match args.kernel {
        Kernel::Old => "old",
        Kernel::Warp => "warp",
    };
    println!("Kernel: {}", kernel_name);

    let db_data = if args.db.starts_with("synthetic:") {
        let size_mb: usize = args.db.strip_prefix("synthetic:").unwrap().parse()?;
        let size_bytes = size_mb * 1024 * 1024;
        println!("Generating synthetic database: {} MB", size_mb);
        io::stdout().flush().unwrap();
        vec![0xABu8; size_bytes]
    } else {
        std::fs::read(&args.db)?
    };
    let num_entries = db_data.len() / args.entry_size;
    println!(
        "Database: {} ({:.2} GB, {} entries)",
        args.db,
        db_data.len() as f64 / 1e9,
        num_entries
    );

    let params = rms24::Params::new(num_entries as u64, args.entry_size, args.lambda);
    println!("Block size: {}", params.block_size);
    println!("Num blocks: {}", params.num_blocks);
    println!("Regular hints: {}", params.num_reg_hints);
    println!("Backup hints: {}", params.num_backup_hints);

    let total_hints = if let Some(max) = args.max_hints {
        max.min(params.total_hints() as u32)
    } else {
        params.total_hints() as u32
    };
    let hint_start = args.hint_start;
    let hint_end = (hint_start + total_hints).min(params.total_hints() as u32);
    let hint_count = hint_end - hint_start;
    println!(
        "Generating: {} hints (range {}..{})",
        hint_count, hint_start, hint_end
    );
    println!();

    #[cfg(feature = "cuda")]
    {
        use rms24::client::Client;
        use rms24::gpu::{GpuHintGenerator, HintMeta, Rms24Params};
        use rms24::hints::{find_median_cutoff, SubsetData};
        use rms24::prf::Prf;

        let gpu_params = Rms24Params {
            num_entries: params.num_entries,
            block_size: params.block_size,
            num_blocks: params.num_blocks,
            num_reg_hints: params.num_reg_hints as u32,
            num_backup_hints: params.num_backup_hints as u32,
            total_hints: hint_count,
            _padding: 0,
        };

        println!("Running Phase 1 (CPU)...");
        io::stdout().flush().unwrap();
        let phase1_start = Instant::now();

        enum Phase1Data {
            Old {
                prf_key: [u32; 8],
                hint_meta: Vec<HintMeta>,
            },
            Warp {
                subset_data: SubsetData,
            },
        }

        let phase1 = match args.kernel {
            Kernel::Old => {
                let prf = Prf::random();
                let prf_key = prf.key_u32();

                let mut hint_meta = Vec::with_capacity(total_hints as usize);
                for hint_idx in 0..total_hints {
                    let mut select_values = prf.select_vector(hint_idx, params.num_blocks as u32);
                    let cutoff = find_median_cutoff(&mut select_values);

                    let (extra_block, extra_offset) =
                        if cutoff > 0 && (hint_idx as u64) < params.num_reg_hints {
                            let mut block = 0u32;
                            for b in 0..params.num_blocks as u32 {
                                if prf.select(hint_idx, b) >= cutoff {
                                    block = b;
                                    break;
                                }
                            }
                            (block, 0u32)
                        } else {
                            (0, 0)
                        };

                    hint_meta.push(HintMeta {
                        cutoff,
                        extra_block,
                        extra_offset,
                        _padding: 0,
                    });
                }
                Phase1Data::Old { prf_key, hint_meta }
            }
            Kernel::Warp => {
                let client = Client::new(params.clone());
                let subsets = client.generate_subsets_range(hint_start as usize, hint_end as usize);
                let subset_data = SubsetData::from_subsets(&subsets);
                Phase1Data::Warp { subset_data }
            }
        };

        let phase1_time = phase1_start.elapsed();
        println!("Phase 1 complete: {:.2}s", phase1_time.as_secs_f64());
        io::stdout().flush().unwrap();

        println!("\nInitializing GPU...");
        io::stdout().flush().unwrap();
        let generator = GpuHintGenerator::new(0)?;

        println!("\nWarming up ({} iterations)...", args.warmup);
        io::stdout().flush().unwrap();
        for _ in 0..args.warmup {
            match &phase1 {
                Phase1Data::Old { prf_key, hint_meta } => {
                    let _ = generator.generate_hints(&db_data, prf_key, hint_meta, gpu_params)?;
                }
                Phase1Data::Warp { subset_data } => {
                    let _ = generator.generate_hints_warp(&db_data, subset_data, gpu_params)?;
                }
            }
        }

        println!("\nBenchmarking ({} iterations)...", args.iterations);
        let mut times = Vec::with_capacity(args.iterations as usize);

        for i in 0..args.iterations {
            let start = Instant::now();
            match &phase1 {
                Phase1Data::Old { prf_key, hint_meta } => {
                    let _ = generator.generate_hints(&db_data, prf_key, hint_meta, gpu_params)?;
                }
                Phase1Data::Warp { subset_data } => {
                    let _ = generator.generate_hints_warp(&db_data, subset_data, gpu_params)?;
                }
            }
            let elapsed = start.elapsed();
            times.push(elapsed.as_millis() as f64);
            println!("Iteration {}: {:.2} ms", i + 1, elapsed.as_millis());
        }

        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        let max = times.iter().cloned().fold(f64::MIN, f64::max);

        println!();
        println!("Results:");
        println!("  Min: {:.2} ms", min);
        println!("  Max: {:.2} ms", max);
        println!("  Avg: {:.2} ms", avg);
        println!(
            "  Throughput: {:.0} hints/sec",
            total_hints as f64 / (avg / 1000.0)
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = args.kernel;
        eprintln!("ERROR: CUDA feature not enabled. Rebuild with --features cuda");
        std::process::exit(1);
    }

    Ok(())
}
