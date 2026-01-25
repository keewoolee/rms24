//! GPU Hint Generation Benchmark for RMS24
//!
//! Usage:
//!   cargo run --release --features cuda --bin bench_gpu_hints -- \
//!     --db data/database.bin --lambda 80 --iterations 5

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "bench_gpu_hints")]
#[command(about = "Benchmark GPU hint generation for RMS24")]
struct Args {
    /// Path to database file
    #[arg(long)]
    db: PathBuf,

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

    /// Entry size in bytes
    #[arg(long, default_value = "40")]
    entry_size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("RMS24 GPU Hint Generation Benchmark");
    println!("====================================");

    let db_data = std::fs::read(&args.db)?;
    let num_entries = db_data.len() / args.entry_size;
    println!(
        "Database: {} ({:.2} GB, {} entries)",
        args.db.display(),
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
    println!("Generating: {} hints", total_hints);
    println!();

    #[cfg(feature = "cuda")]
    {
        use rms24::gpu::{GpuHintGenerator, HintMeta, Rms24Params};
        use rms24::hints::find_median_cutoff;
        use rms24::prf::Prf;

        // Phase 1 on CPU first (GPU is expensive to initialize)
        println!("Running Phase 1 (CPU)...");
        let phase1_start = Instant::now();

        let prf = Prf::random();
        let prf_key = prf.key_u32();

        let gpu_params = Rms24Params {
            num_entries: params.num_entries,
            block_size: params.block_size,
            num_blocks: params.num_blocks,
            num_reg_hints: params.num_reg_hints as u32,
            num_backup_hints: params.num_backup_hints as u32,
            total_hints,
            _padding: 0,
        };

        let mut hint_meta = Vec::with_capacity(total_hints as usize);
        for hint_idx in 0..total_hints {
            let select_values = prf.select_vector(hint_idx, params.num_blocks as u32);
            let cutoff = find_median_cutoff(&select_values);

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

        let phase1_time = phase1_start.elapsed();
        println!("Phase 1 complete: {:.2}s", phase1_time.as_secs_f64());

        // Initialize GPU after CPU work
        println!("\nInitializing GPU...");
        let generator = GpuHintGenerator::new(0)?;

        println!("\nWarming up ({} iterations)...", args.warmup);
        for _ in 0..args.warmup {
            let _ = generator.generate_hints(&db_data, &prf_key, &hint_meta, gpu_params)?;
        }

        println!("\nBenchmarking ({} iterations)...", args.iterations);
        let mut times = Vec::with_capacity(args.iterations as usize);

        for i in 0..args.iterations {
            let start = Instant::now();
            let _ = generator.generate_hints(&db_data, &prf_key, &hint_meta, gpu_params)?;
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
        eprintln!("ERROR: CUDA feature not enabled. Rebuild with --features cuda");
        std::process::exit(1);
    }

    Ok(())
}
