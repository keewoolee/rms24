//! Integration tests for hint generation.

use rms24::{client::Client, params::Params};

#[test]
fn test_cpu_hint_generation_roundtrip() {
    let num_entries = 1000u64;
    let entry_size = 40;
    let security_param = 4;

    let params = Params::new(num_entries, entry_size, security_param);
    let mut client = Client::new(params.clone());

    // Create test database with unique entries
    let mut db = vec![0u8; num_entries as usize * entry_size];
    for i in 0..num_entries as usize {
        // Each entry has unique first byte
        db[i * entry_size] = (i % 256) as u8;
        db[i * entry_size + 1] = ((i / 256) % 256) as u8;
        // TAG in last 8 bytes
        db[i * entry_size + 32..i * entry_size + 40].copy_from_slice(&(i as u64).to_le_bytes());
    }

    client.generate_hints(&db);

    // Verify hints were generated
    let valid_hints = client.hints.cutoffs.iter().filter(|&&c| c > 0).count();
    assert!(valid_hints > 0, "Should have valid hints");

    // Verify parities are not all zero (statistically unlikely with real data)
    let nonzero_parities = client
        .hints
        .parities
        .iter()
        .filter(|p| p.iter().any(|&b| b != 0))
        .count();
    assert!(nonzero_parities > 0, "Should have non-zero parities");
}

#[test]
fn test_hint_coverage() {
    let params = Params::new(100, 40, 8);
    let mut client = Client::new(params.clone());
    let db = vec![0xFFu8; 100 * 40];

    client.generate_hints(&db);

    // With security_param=8 and block_size=10, we should have 80 regular hints
    assert_eq!(params.num_reg_hints, 80);

    // Most hints should be valid (cutoff > 0)
    let num_reg = params.num_reg_hints as usize;
    let valid_count = client.hints.cutoffs[..num_reg].iter().filter(|&&c| c > 0).count();
    assert!(valid_count >= 75, "Most regular hints should be valid, got {}", valid_count);
}

#[test]
fn test_backup_hints() {
    let params = Params::new(256, 40, 4);
    let mut client = Client::new(params.clone());
    
    // Database with pattern
    let mut db = vec![0u8; 256 * 40];
    for i in 0..256 {
        for j in 0..40 {
            db[i * 40 + j] = (i ^ j) as u8;
        }
    }

    client.generate_hints(&db);

    // Check backup hints have dual parities
    let num_backup = params.num_backup_hints as usize;
    assert_eq!(client.hints.backup_parities_high.len(), num_backup);
    
    // Some backup hints should have non-zero high parities
    let nonzero_high = client.hints.backup_parities_high
        .iter()
        .filter(|p| p.iter().any(|&b| b != 0))
        .count();
    assert!(nonzero_high > 0, "Backup hints should have non-zero high parities");
}

#[test]
fn test_deterministic_with_same_prf() {
    let params = Params::new(100, 40, 4);
    let prf_key = [0u8; 32];
    
    let mut client1 = Client::with_prf(params.clone(), rms24::prf::HmacPrf::new(prf_key));
    let mut client2 = Client::with_prf(params, rms24::prf::HmacPrf::new(prf_key));
    
    let db = vec![0x42u8; 100 * 40];
    
    client1.generate_hints(&db);
    client2.generate_hints(&db);
    
    // Cutoffs should be identical
    assert_eq!(client1.hints.cutoffs, client2.hints.cutoffs);
    
    // Note: Parities may differ due to random extra selection,
    // but cutoffs are deterministic from PRF
}

#[test]
fn test_large_database() {
    // Test with larger database to ensure no panics
    let num_entries = 10_000u64;
    let params = Params::new(num_entries, 40, 4);
    let mut client = Client::new(params);
    
    let db = vec![0u8; num_entries as usize * 40];
    client.generate_hints(&db);
    
    let valid_hints = client.hints.cutoffs.iter().filter(|&&c| c > 0).count();
    assert!(valid_hints > 0);
}
