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
