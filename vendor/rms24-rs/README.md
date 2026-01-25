# RMS24-RS

Rust implementation of RMS24 single-server PIR with CUDA-accelerated hint generation.

## Features

- RMS24 protocol with median-cutoff subset selection
- 8-byte TAG fingerprints for keyword PIR (Plinko-style)
- CUDA acceleration for hint generation (optional `cuda` feature)

## Building

```bash
# CPU only
cargo build --release

# With CUDA
cargo build --release --features cuda
```

## Entry Format

40 bytes per entry:
- 32 bytes: value
- 8 bytes: TAG (keccak256 fingerprint)
