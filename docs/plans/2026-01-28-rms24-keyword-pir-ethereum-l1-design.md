# RMS24 Keyword PIR for Ethereum L1 (Schema40)

Date: 2026-01-28

## Summary

Implement a Rust RMS24 keyword PIR stack for Ethereum L1 data using the 40-byte schema from `plinko-rs`, while keeping RMS24 core and keyword PIR as separate modules. The keyword layer uses cuckoo hashing over fixed-size 40B entries and an 8B tag (keccak256 fingerprint). To avoid increasing entry size, we keep the 8B tag and add a small collision side table for the rare cases where tags collide. Hint generation runs inside a TEE; the client can optionally run fully inside the TEE.

## Goals

- Port `schema40` semantics (account/storage entries, tag derivation, code store) into `vendor/rms24-rs`.
- Implement RMS24 online protocol (query/answer/extract/replenish/updates) as index PIR.
- Add a keyword PIR module that wraps RMS24 using cuckoo hashing and tag verification.
- Support both account keys (address) and storage keys (address || slot_key).
- Add an integration test path against a real-data slice derived from `/mnt/mainnet-pir-data-v3`.

## Non-Goals

- Redesign RMS24 protocol or its parameters.
- Replace the 40B schema with a larger entry size.
- Provide a production-grade TEE deployment stack (document behavior only).

## Architecture

Modules in `vendor/rms24-rs`:

- `rms24` core: `params`, `prf`, `hints`, `client`, `server`, `messages`, `updates`.
- `schema40`: `AccountEntry40`, `StorageEntry40`, `Tag`, `CodeStore` (ported from `plinko-rs`).
- `keyword_pir`: cuckoo hashing + client/server wrappers built on RMS24 index PIR.

The RMS24 core treats the database as a flat array of fixed-size entries. The keyword layer maps full keys to cuckoo positions, issues k index PIR queries, verifies the tag, and returns the decoded entry.

## Data Flow

Build time (server side):

1) Read Ethereum L1 artifacts (`database.bin`, `account-mapping.bin`, `storage-mapping.bin`).
2) Derive keys:
   - Account key = 20B address.
   - Storage key = 20B address || 32B slot_key.
3) Compute tag = keccak256(key)[0:8].
4) Encode 40B entries using `schema40` and insert into a cuckoo table.
5) Detect tag collisions and build a small collision side table.

Client setup (TEE):

- Stream the RMS24 DB snapshot into the TEE to generate hints; discard the raw DB afterward.
- Store the collision tag set and mapping files for updates.

Online query:

- Compute cuckoo positions and issue k RMS24 queries to the main table.
- Verify the tag. If the tag is in the collision set, issue one additional keyword PIR query to the collision table to disambiguate (privacy leak accepted as rare).

## Collision Handling

We keep 40B entries and the 8B tag. To avoid increasing entry size, we introduce a collision side table populated only with keys whose tags collide. The client holds a compact collision tag set and only queries the side table when needed. This preserves storage/bandwidth while bounding collision errors to a controlled fallback path.

## TEE Assumptions

- Offline hint generation runs inside a TEE.
- The protocol can be extended so the client runs fully inside the TEE (offline + online); document this as optional.

## Testing & Validation

Unit tests:

- `schema40` encode/decode and tag determinism.
- RMS24 PRF, cutoff, and hint state invariants.
- Keyword PIR cuckoo placement + lookup correctness.

Integration tests:

- Optional real-data tests run when `RMS24_DATA_DIR` is set.
- Use a 1,000,000-entry slice (40MB) from `/mnt/mainnet-pir-data-v3/database.bin` and filtered mapping slices (indices < N).
- Verify account and storage queries against the slice; ensure tag verification matches expected entries.

## Data Slice & Distribution

A slice generator will:

- Extract the first N entries from `database.bin`.
- Filter `account-mapping.bin` and `storage-mapping.bin` to indices < N.
- Write `metadata.json` with sizes and checksums.
- Upload artifacts to R2 bucket `pir` with public URLs under `https://pir.53627.org/`.

A download helper script will pull these artifacts for tests and verify checksums.

## Open Questions

None at this stage. Follow-up implementation planning should decide exact module APIs and test harness integration.
