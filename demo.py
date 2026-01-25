#!/usr/bin/env python3
"""
Demo and benchmarks for RMS24 PIR and Keyword PIR.

Default: 32MB database with detailed timing and communication metrics.

Usage:
    python3 demo.py              # Run RMS24 demo (32MB DB)
    python3 demo.py --kpir       # Run Keyword PIR demo
    python3 demo.py --all        # Run both demos
    python3 demo.py --small      # Run with smaller database (faster)
"""

import argparse
import time

from pir.rms24 import Params, Client, Server
from pir.keyword_pir import KPIRParams, KPIRClient, KPIRServer


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_count(n: int) -> str:
    """Format number with K/M suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def format_bytes(n: int) -> str:
    """Format bytes with KiB/MiB/GiB suffix."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.2f} GiB"
    if n >= 1024 * 1024:
        return f"{n / (1024**2):.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def format_time(seconds: float) -> str:
    """Format time with appropriate unit."""
    if seconds >= 1:
        return f"{seconds:.2f}s"
    if seconds >= 0.001:
        return f"{seconds*1000:.2f}ms"
    return f"{seconds*1_000_000:.1f}us"


# =============================================================================
# RMS24 PIR Demo
# =============================================================================


def create_database(num_entries: int, entry_size: int) -> list[bytes]:
    """Create a database with sequential values."""
    return [
        i.to_bytes(min(entry_size, 8), "little").ljust(entry_size, b"\x00")
        for i in range(num_entries)
    ]


def compute_query_size(params: Params) -> int:
    """Compute size of a single query in bytes."""
    mask_size = (params.num_blocks + 7) // 8
    # offsets: num_blocks/2 integers, assume 4 bytes each (could be 2 for small DBs)
    offsets_size = (params.num_blocks // 2) * 4
    return mask_size + offsets_size


def compute_response_size(params: Params) -> int:
    """Compute size of a single response in bytes."""
    return 2 * params.entry_size  # parity_0 + parity_1


def run_rms24_demo(num_entries: int, entry_size: int, security_param: int, num_queries: int):
    """Run the RMS24 PIR demo with detailed metrics."""
    print("=" * 70)
    print("RMS24 Single-Server PIR - Demo & Benchmarks")
    print("=" * 70)

    # Setup parameters
    params = Params(
        num_entries=num_entries,
        entry_size=entry_size,
        security_param=security_param,
    )

    db_size = num_entries * entry_size
    query_size = compute_query_size(params)
    response_size = compute_response_size(params)

    print(f"\n{'Parameters':─^70}")
    print(f"  Database size:      {format_bytes(db_size):>12}  ({format_count(num_entries)} × {entry_size}B)")
    print(f"  Block size (w):     {params.block_size:>12}")
    print(f"  Num blocks (c):     {params.num_blocks:>12}")
    print(f"  Regular hints (M):  {format_count(params.num_reg_hints):>12}")
    print(f"  Backup hints:       {format_count(params.num_backup_hints):>12}")
    print(f"  Security param (λ): {security_param:>12}")

    # Create database
    print(f"\n{'Database Creation':─^70}")
    start = time.perf_counter()
    db = create_database(num_entries, entry_size)
    db_time = time.perf_counter() - start
    print(f"  Time: {format_time(db_time)}")

    server = Server(db, params)
    client = Client(params)

    # Offline phase
    print(f"\n{'Offline Phase (Hint Generation)':─^70}")
    start = time.perf_counter()
    client.generate_hints(server.stream_database())
    offline_time = time.perf_counter() - start

    print(f"  Time:           {format_time(offline_time):>12}")
    print(f"  Throughput:     {num_entries / offline_time:>12,.0f} entries/s")
    print(f"  Download:       {format_bytes(db_size):>12}  (full database)")

    # Online phase
    num_queries = min(num_queries, client.remaining_queries())
    print(f"\n{'Online Phase (' + str(num_queries) + ' queries)':─^70}")

    test_indices = [(i * num_entries) // num_queries for i in range(num_queries)]

    # Detailed timing breakdown
    query_gen_times = []
    server_answer_times = []
    extract_times = []
    replenish_times = []
    all_correct = True

    for idx in test_indices:
        # Query generation (client)
        start = time.perf_counter()
        queries = client.query([idx])
        query_gen_times.append(time.perf_counter() - start)

        # Server answer
        start = time.perf_counter()
        responses = server.answer(queries)
        server_answer_times.append(time.perf_counter() - start)

        # Extract result (client)
        start = time.perf_counter()
        [result] = client.extract(responses)
        extract_times.append(time.perf_counter() - start)

        # Replenish hints (client)
        start = time.perf_counter()
        client.replenish_hints()
        replenish_times.append(time.perf_counter() - start)

        if result != db[idx]:
            all_correct = False
            print(f"  ERROR: index {idx} returned wrong result!")

    avg_query_gen = sum(query_gen_times) / len(query_gen_times)
    avg_server = sum(server_answer_times) / len(server_answer_times)
    avg_extract = sum(extract_times) / len(extract_times)
    avg_replenish = sum(replenish_times) / len(replenish_times)
    avg_total = avg_query_gen + avg_server + avg_extract + avg_replenish

    print(f"  Correctness:    {'PASS' if all_correct else 'FAIL':>12}")
    print(f"\n  Timing breakdown (avg per query):")
    print(f"    Client query():      {format_time(avg_query_gen):>10}")
    print(f"    Server answer():     {format_time(avg_server):>10}")
    print(f"    Client extract():    {format_time(avg_extract):>10}")
    print(f"    Client replenish():  {format_time(avg_replenish):>10}")
    print(f"    ─────────────────────────────────")
    print(f"    Total:               {format_time(avg_total):>10}")

    # Communication costs
    print(f"\n{'Communication Costs':─^70}")
    print(f"  Offline download:   {format_bytes(db_size):>12}")
    print(f"  Query size:         {format_bytes(query_size):>12}  (mask + offsets)")
    print(f"  Response size:      {format_bytes(response_size):>12}  (2 × entry_size)")
    print(f"  Online total:       {format_bytes(query_size + response_size):>12}  per query")

    # Update phase
    print(f"\n{'Update Phase (10 updates)':─^70}")
    update_server_times = []
    update_client_times = []

    for i in range(10):
        idx = (i * num_entries) // 10
        new_value = bytes([i] * entry_size)

        start = time.perf_counter()
        updates = server.update_entries({idx: new_value})
        update_server_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        client.update_hints(updates)
        update_client_times.append(time.perf_counter() - start)

    avg_server_update = sum(update_server_times) / len(update_server_times)
    avg_client_update = sum(update_client_times) / len(update_client_times)
    update_msg_size = 4 + entry_size  # index (4 bytes) + delta

    print(f"  Timing breakdown (avg per update):")
    print(f"    Server update_entries(): {format_time(avg_server_update):>10}")
    print(f"    Client update_hints():   {format_time(avg_client_update):>10}")
    print(f"  Update message size:   {format_bytes(update_msg_size):>12}  (index + delta)")

    # Summary
    print(f"\n{'Summary':─^70}")
    print(f"  Offline:  {format_time(offline_time)} to download {format_bytes(db_size)}")
    print(f"  Online:   {format_time(avg_total)}/query, {format_bytes(query_size + response_size)} communication")
    print(f"  Capacity: {client.remaining_queries()} queries remaining")
    print("=" * 70)


# =============================================================================
# Keyword PIR Demo
# =============================================================================


def create_kv_store(num_items: int, key_size: int, value_size: int) -> dict[bytes, bytes]:
    """Create a key-value store with sequential keys and values."""
    return {
        f"key_{i:0{key_size-4}d}".encode()[:key_size]: f"val_{i:0{value_size-4}d}".encode()[:value_size]
        for i in range(num_items)
    }


def run_kpir_demo(num_items: int, key_size: int, value_size: int, security_param: int, num_queries: int):
    """Run the Keyword PIR demo with detailed metrics."""
    print("=" * 70)
    print("Keyword PIR (KPIR) - Demo & Benchmarks")
    print("=" * 70)

    # Setup parameters
    kw_params = KPIRParams(
        num_items_expected=num_items,
        key_size=key_size,
        value_size=value_size,
    )

    pir_params = Params(
        num_entries=kw_params.num_buckets,
        entry_size=kw_params.entry_size,
        security_param=security_param,
    )

    logical_db_size = num_items * (key_size + value_size)
    pir_db_size = kw_params.num_buckets * kw_params.entry_size
    query_size = compute_query_size(pir_params) * kw_params.num_hashes
    response_size = compute_response_size(pir_params) * kw_params.num_hashes

    print(f"\n{'Parameters':─^70}")
    print(f"  Logical DB size:    {format_bytes(logical_db_size):>12}  ({format_count(num_items)} items)")
    print(f"  Key size:           {key_size:>12} bytes")
    print(f"  Value size:         {value_size:>12} bytes")
    print(f"  Cuckoo buckets:     {format_count(kw_params.num_buckets):>12}")
    print(f"  Cuckoo hashes:      {kw_params.num_hashes:>12}")
    print(f"  PIR entry size:     {kw_params.entry_size:>12} bytes")
    print(f"  PIR DB size:        {format_bytes(pir_db_size):>12}")

    # Create key-value store
    print(f"\n{'Key-Value Store Creation':─^70}")
    start = time.perf_counter()
    kv_store = create_kv_store(num_items, key_size, value_size)
    kv_time = time.perf_counter() - start
    print(f"  Time: {format_time(kv_time)}")

    # Build cuckoo table
    print(f"\n{'Cuckoo Table Construction':─^70}")
    start = time.perf_counter()
    server = KPIRServer.create(kv_store, kw_params, Server, security_param)
    build_time = time.perf_counter() - start
    print(f"  Time: {format_time(build_time)}")

    # Create client
    client = KPIRClient.create(kw_params, Client, security_param)

    # Offline phase
    print(f"\n{'Offline Phase (Hint Generation)':─^70}")
    start = time.perf_counter()
    client.generate_hints(server.stream_database())
    offline_time = time.perf_counter() - start

    print(f"  Time:           {format_time(offline_time):>12}")
    print(f"  Download:       {format_bytes(pir_db_size):>12}  (PIR database)")

    # Online phase
    num_queries = min(num_queries, client.remaining_queries())
    print(f"\n{'Online Phase (' + str(num_queries) + ' keyword queries)':─^70}")

    test_keys = list(kv_store.keys())[:num_queries]

    # Detailed timing breakdown
    query_gen_times = []
    server_answer_times = []
    extract_times = []
    replenish_times = []
    all_correct = True

    for key in test_keys:
        # Query generation (client)
        start = time.perf_counter()
        queries = client.query([key])
        query_gen_times.append(time.perf_counter() - start)

        # Server answer
        start = time.perf_counter()
        responses, stash = server.answer(queries)
        server_answer_times.append(time.perf_counter() - start)

        # Extract result (client)
        start = time.perf_counter()
        [result] = client.extract(responses, stash)
        extract_times.append(time.perf_counter() - start)

        # Replenish hints (client)
        start = time.perf_counter()
        client.replenish_hints()
        replenish_times.append(time.perf_counter() - start)

        expected = kv_store[key]
        if result != expected:
            all_correct = False
            print(f"  ERROR: key {key} returned wrong value!")

    avg_query_gen = sum(query_gen_times) / len(query_gen_times)
    avg_server = sum(server_answer_times) / len(server_answer_times)
    avg_extract = sum(extract_times) / len(extract_times)
    avg_replenish = sum(replenish_times) / len(replenish_times)
    avg_total = avg_query_gen + avg_server + avg_extract + avg_replenish

    print(f"  Correctness:    {'PASS' if all_correct else 'FAIL':>12}")
    print(f"\n  Timing breakdown (avg per query):")
    print(f"    Client query():      {format_time(avg_query_gen):>10}  ({kw_params.num_hashes} PIR queries)")
    print(f"    Server answer():     {format_time(avg_server):>10}")
    print(f"    Client extract():    {format_time(avg_extract):>10}")
    print(f"    Client replenish():  {format_time(avg_replenish):>10}")
    print(f"    ─────────────────────────────────")
    print(f"    Total:               {format_time(avg_total):>10}")

    # Communication costs
    print(f"\n{'Communication Costs':─^70}")
    print(f"  Offline download:   {format_bytes(pir_db_size):>12}")
    print(f"  Query size:         {format_bytes(query_size):>12}  ({kw_params.num_hashes} PIR queries)")
    print(f"  Response size:      {format_bytes(response_size):>12}  ({kw_params.num_hashes} PIR responses)")
    print(f"  Online total:       {format_bytes(query_size + response_size):>12}  per keyword query")

    # Update phase
    print(f"\n{'Update Phase (10 updates)':─^70}")
    update_server_times = []
    update_client_times = []

    update_keys = list(kv_store.keys())[:10]
    for i, key in enumerate(update_keys):
        new_value = f"updated-{i}".encode().ljust(value_size, b"\x00")

        start = time.perf_counter()
        updates = server.update({key: new_value})
        update_server_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        client.update_hints(updates)
        update_client_times.append(time.perf_counter() - start)

    avg_server_update = sum(update_server_times) / len(update_server_times)
    avg_client_update = sum(update_client_times) / len(update_client_times)
    update_msg_size = 4 + kw_params.entry_size  # index (4 bytes) + delta

    print(f"  Timing breakdown (avg per update):")
    print(f"    Server update():         {format_time(avg_server_update):>10}")
    print(f"    Client update_hints():   {format_time(avg_client_update):>10}")
    print(f"  Update message size:   {format_bytes(update_msg_size):>12}  (index + delta)")

    # Summary
    print(f"\n{'Summary':─^70}")
    print(f"  Offline:  {format_time(offline_time)} to download {format_bytes(pir_db_size)}")
    print(f"  Online:   {format_time(avg_total)}/query, {format_bytes(query_size + response_size)} communication")
    print(f"  Capacity: {client.remaining_queries()} queries remaining")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


# Default: 1MB database with 32-byte entries = 32768 entries
DEFAULT_NUM_ENTRIES = 32768
DEFAULT_ENTRY_SIZE = 32
DEFAULT_KEY_SIZE = 32
DEFAULT_SECURITY_PARAM = 80
DEFAULT_NUM_QUERIES = 10


def main():
    parser = argparse.ArgumentParser(
        description="RMS24 PIR & KPIR Demo with detailed benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 demo.py                    # RMS24 with 32MB database
  python3 demo.py --kpir             # Keyword PIR with 32MB database
  python3 demo.py --all              # Both demos
  python3 demo.py --small            # Quick test with 1MB database
  python3 demo.py --entries 1000000  # Custom 1M entries
        """,
    )
    parser.add_argument("--kpir", action="store_true", help="Run Keyword PIR demo")
    parser.add_argument("--all", action="store_true", help="Run both demos")
    parser.add_argument("--small", action="store_true", help="Use smaller database (~1MB, faster)")
    parser.add_argument("--entries", type=int, default=None, help="Number of entries/items")
    parser.add_argument("--entry-size", type=int, default=DEFAULT_ENTRY_SIZE, help=f"Entry/value size (default: {DEFAULT_ENTRY_SIZE})")
    parser.add_argument("--key-size", type=int, default=DEFAULT_KEY_SIZE, help=f"Key size for KPIR (default: {DEFAULT_KEY_SIZE})")
    parser.add_argument("--security", type=int, default=DEFAULT_SECURITY_PARAM, help=f"Security parameter (default: {DEFAULT_SECURITY_PARAM})")
    parser.add_argument("--queries", type=int, default=DEFAULT_NUM_QUERIES, help=f"Number of queries (default: {DEFAULT_NUM_QUERIES})")
    args = parser.parse_args()

    # Determine number of entries
    if args.entries:
        num_entries = args.entries
    elif args.small:
        num_entries = 32768  # ~1MB with 32-byte entries
    else:
        num_entries = DEFAULT_NUM_ENTRIES  # 32MB with 32-byte entries

    # Run demos
    if args.all:
        run_rms24_demo(num_entries, args.entry_size, args.security, args.queries)
        print("\n\n")
        run_kpir_demo(num_entries, args.key_size, args.entry_size, args.security, args.queries)
    elif args.kpir:
        run_kpir_demo(num_entries, args.key_size, args.entry_size, args.security, args.queries)
    else:
        run_rms24_demo(num_entries, args.entry_size, args.security, args.queries)


if __name__ == "__main__":
    main()
