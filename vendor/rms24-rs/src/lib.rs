//! RMS24 single-server PIR implementation.
//!
//! Based on "Simple and Practical Amortized Sublinear Private Information
//! Retrieval" (https://eprint.iacr.org/2024/1362).

pub mod params;
pub mod prf;
pub mod hints;
pub mod client;
// pub mod server;

// #[cfg(feature = "cuda")]
// pub mod gpu;

pub use params::Params;
