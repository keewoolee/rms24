"""
Run RMS24 GPU benchmark on Modal.

Usage:
    modal run scripts/modal_run_bench.py --gpu h200

Based on Plinko's modal_run_bench.py, adapted for RMS24 protocol.
"""

import modal

app = modal.App("rms24-bench")
volume = modal.Volume.from_name("plinko-data", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=["target", ".git", ".jj", "__pycache__", ".DS_Store", "data"],
    )
)


def _run_bench(
    gpu: str, db_path: str, lambda_param: int, iterations: int, max_hints: int = None
) -> dict:
    import os
    import subprocess

    os.chdir("/app")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    size_gb = os.path.getsize(db_path) / 1e9
    print(f"Data: {db_path} ({size_gb:.2f} GB)")

    env = os.environ.copy()
    env["PATH"] = f"/root/.cargo/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_ROOT"] = "/usr/local/cuda"
    env["CUDA_PATH"] = "/usr/local/cuda"
    env["CUDA_ARCH"] = "sm_90"

    print("\n=== Building ===")
    build_result = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--bin",
            "bench_gpu_hints",
            "--features",
            "cuda",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if build_result.returncode != 0:
        print(build_result.stderr[-3000:])
        raise RuntimeError("Build failed")
    print("Build succeeded")

    print(f"\n=== Benchmarking on {gpu} ===")
    cmd = [
        "./target/release/bench_gpu_hints",
        "--db",
        db_path,
        "--lambda",
        str(lambda_param),
        "--iterations",
        str(iterations),
        "--warmup",
        "2",
    ]
    if max_hints:
        cmd.extend(["--max-hints", str(max_hints)])

    import sys

    sys.stdout.flush()
    result = subprocess.run(cmd, env=env)

    return {"gpu": gpu, "output": f"Exit code: {result.returncode}"}


@app.function(image=image, gpu="H200", volumes={"/data": volume}, timeout=7200)
def bench_h200(
    db_path: str,
    lambda_param: int = 80,
    iterations: int = 5,
    max_hints: int = None,
):
    return _run_bench("H200", db_path, lambda_param, iterations, max_hints)


@app.function(image=image, gpu="B200", volumes={"/data": volume}, timeout=7200)
def bench_b200(
    db_path: str,
    lambda_param: int = 80,
    iterations: int = 5,
    max_hints: int = None,
):
    return _run_bench("B200", db_path, lambda_param, iterations, max_hints)


@app.local_entrypoint()
def main(
    gpu: str = "h200",
    db: str = "/data/mainnet-v3/database.bin",
    lambda_param: int = 80,
    iterations: int = 5,
    max_hints: int = None,
):
    """
    Run RMS24 GPU benchmark.

    Args:
        gpu: GPU type (h200, b200)
        db: Path to database file on Modal volume
        lambda_param: Security parameter
        iterations: Number of benchmark iterations
        max_hints: Maximum hints to generate
    """
    print("=== RMS24 GPU BENCHMARK ===")
    print(f"GPU: {gpu.upper()}")
    print(f"Database: {db}")
    print(f"Lambda: {lambda_param}")
    print(f"Iterations: {iterations}")
    if max_hints:
        print(f"Max hints: {max_hints:,}")
    print()

    if gpu.lower() == "h200":
        result = bench_h200.remote(db, lambda_param, iterations, max_hints)
    elif gpu.lower() == "b200":
        result = bench_b200.remote(db, lambda_param, iterations, max_hints)
    else:
        raise ValueError(f"Unknown GPU: {gpu}. Supported: h200, b200")

    print("\n" + "=" * 50)
    print(result["output"])
