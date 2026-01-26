"""
Correctness tests for the PyTorch hint generation kernel.

Verifies that:
1. XOR reduction is correct
2. Masking works properly (invalid entries don't contribute)
3. Edge cases are handled
"""

import torch
import unittest
import sys
import os

# Support testing different kernel implementations
KERNEL_MODULE = os.environ.get("KERNEL_MODULE", "forge_minimal")

if KERNEL_MODULE == "forge_optimized":
    from forge_optimized import CUDAModel as HintGenKernel, PytorchModel
elif KERNEL_MODULE == "forge_v2":
    from forge_v2 import HintGenKernel
else:
    from forge_minimal import HintGenKernel


class TestHintGenKernelCorrectness(unittest.TestCase):
    
    def setUp(self):
        self.device = "cpu"  # Use CPU for deterministic testing
        self.kernel = HintGenKernel()
    
    def test_single_entry_per_hint(self):
        """Each hint gathers exactly one entry - parity should equal that entry."""
        num_entries = 100
        num_hints = 10
        max_subset_size = 5
        
        # Create entries with known values
        entries = torch.arange(num_entries * 5, dtype=torch.int64).view(num_entries, 5)
        
        # Each hint picks one unique entry
        padded_indices = torch.zeros(num_hints, max_subset_size, dtype=torch.int64)
        for i in range(num_hints):
            padded_indices[i, 0] = i * 10  # Entry 0, 10, 20, ...
        
        # Only first index is valid
        valid_mask = torch.zeros(num_hints, max_subset_size, dtype=torch.bool)
        valid_mask[:, 0] = True
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        # Each parity should equal the single gathered entry
        for i in range(num_hints):
            expected = entries[i * 10]
            self.assertTrue(torch.equal(parities[i], expected),
                f"Hint {i}: expected {expected}, got {parities[i]}")
    
    def test_xor_two_entries(self):
        """XOR of two entries should be correct."""
        entries = torch.tensor([
            [0xFF, 0x00, 0xAA, 0x55, 0x12],
            [0x0F, 0xF0, 0x55, 0xAA, 0x21],
        ], dtype=torch.int64)
        
        expected_xor = torch.tensor([
            0xFF ^ 0x0F, 0x00 ^ 0xF0, 0xAA ^ 0x55, 0x55 ^ 0xAA, 0x12 ^ 0x21
        ], dtype=torch.int64)
        
        padded_indices = torch.tensor([[0, 1, 0]], dtype=torch.int64)  # indices 0, 1
        valid_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        self.assertTrue(torch.equal(parities[0], expected_xor),
            f"Expected {expected_xor}, got {parities[0]}")
    
    def test_xor_self_is_zero(self):
        """XOR of an entry with itself should be zero."""
        entries = torch.tensor([
            [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x87654321, 0xFFFFFFFF],
        ], dtype=torch.int64)
        
        # Same index twice
        padded_indices = torch.tensor([[0, 0, 0]], dtype=torch.int64)
        valid_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        expected = torch.zeros(5, dtype=torch.int64)
        self.assertTrue(torch.equal(parities[0], expected),
            f"XOR with self should be 0, got {parities[0]}")
    
    def test_invalid_entries_ignored(self):
        """Invalid (masked) entries should not affect parity."""
        entries = torch.tensor([
            [1, 2, 3, 4, 5],
            [100, 200, 300, 400, 500],  # This should be ignored
            [10, 20, 30, 40, 50],
        ], dtype=torch.int64)
        
        padded_indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        valid_mask = torch.tensor([[True, False, True]], dtype=torch.bool)  # Middle one invalid
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        # Should be XOR of entries 0 and 2 only
        expected = entries[0] ^ entries[2]
        self.assertTrue(torch.equal(parities[0], expected),
            f"Expected {expected}, got {parities[0]}")
    
    def test_all_invalid_gives_zero(self):
        """All invalid mask should give zero parity."""
        entries = torch.randint(1, 1000, (10, 5), dtype=torch.int64)
        
        padded_indices = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
        valid_mask = torch.zeros(1, 5, dtype=torch.bool)
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        expected = torch.zeros(5, dtype=torch.int64)
        self.assertTrue(torch.equal(parities[0], expected),
            f"All invalid should give 0, got {parities[0]}")
    
    def test_multiple_hints_independent(self):
        """Multiple hints should be computed independently."""
        entries = torch.arange(50, dtype=torch.int64).view(10, 5)
        
        # Hint 0: entries 0, 1 -> XOR
        # Hint 1: entries 2, 3 -> XOR  
        # Hint 2: entry 4 only
        padded_indices = torch.tensor([
            [0, 1, 0],
            [2, 3, 0],
            [4, 0, 0],
        ], dtype=torch.int64)
        
        valid_mask = torch.tensor([
            [True, True, False],
            [True, True, False],
            [True, False, False],
        ], dtype=torch.bool)
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        expected_0 = entries[0] ^ entries[1]
        expected_1 = entries[2] ^ entries[3]
        expected_2 = entries[4]
        
        self.assertTrue(torch.equal(parities[0], expected_0))
        self.assertTrue(torch.equal(parities[1], expected_1))
        self.assertTrue(torch.equal(parities[2], expected_2))
    
    def test_large_xor_chain(self):
        """XOR of many entries should be correct."""
        num_entries = 1000
        chain_length = 100
        
        entries = torch.randint(0, 2**60, (num_entries, 5), dtype=torch.int64)
        
        # Pick first 100 entries
        padded_indices = torch.arange(chain_length, dtype=torch.int64).unsqueeze(0)
        valid_mask = torch.ones(1, chain_length, dtype=torch.bool)
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        # Compute expected via loop
        expected = entries[0].clone()
        for i in range(1, chain_length):
            expected ^= entries[i]
        
        self.assertTrue(torch.equal(parities[0], expected),
            f"Large chain XOR mismatch")
    
    def test_associativity(self):
        """XOR should be associative - order shouldn't matter."""
        entries = torch.randint(0, 2**60, (10, 5), dtype=torch.int64)
        
        # Same entries in different order
        padded_indices_1 = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
        padded_indices_2 = torch.tensor([[4, 3, 2, 1, 0]], dtype=torch.int64)
        valid_mask = torch.ones(1, 5, dtype=torch.bool)
        
        parities_1 = self.kernel(entries, padded_indices_1, valid_mask)
        parities_2 = self.kernel(entries, padded_indices_2, valid_mask)
        
        self.assertTrue(torch.equal(parities_1, parities_2),
            "XOR should be associative/commutative")
    
    def test_reference_implementation(self):
        """Compare against simple reference implementation."""
        num_entries = 500
        num_hints = 20
        max_subset_size = 50
        
        torch.manual_seed(42)
        entries = torch.randint(0, 2**60, (num_entries, 5), dtype=torch.int64)
        padded_indices = torch.randint(0, num_entries, (num_hints, max_subset_size), dtype=torch.int64)
        valid_mask = torch.rand(num_hints, max_subset_size) < 0.7
        
        # Kernel result
        kernel_result = self.kernel(entries, padded_indices, valid_mask)
        
        # Reference: simple loop
        ref_result = torch.zeros(num_hints, 5, dtype=torch.int64)
        for h in range(num_hints):
            parity = torch.zeros(5, dtype=torch.int64)
            for i in range(max_subset_size):
                if valid_mask[h, i]:
                    parity ^= entries[padded_indices[h, i]]
            ref_result[h] = parity
        
        self.assertTrue(torch.equal(kernel_result, ref_result),
            "Kernel result should match reference implementation")


class TestEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.kernel = HintGenKernel()
    
    def test_empty_hints(self):
        """Zero hints should return empty tensor."""
        entries = torch.randint(0, 100, (10, 5), dtype=torch.int64)
        padded_indices = torch.zeros(0, 5, dtype=torch.int64)
        valid_mask = torch.zeros(0, 5, dtype=torch.bool)
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        self.assertEqual(parities.shape, (0, 5))
    
    def test_max_int64_values(self):
        """Should handle large int64 values correctly."""
        max_val = 2**63 - 1
        entries = torch.tensor([
            [max_val, max_val, max_val, max_val, max_val],
            [1, 1, 1, 1, 1],
        ], dtype=torch.int64)
        
        padded_indices = torch.tensor([[0, 1]], dtype=torch.int64)
        valid_mask = torch.tensor([[True, True]], dtype=torch.bool)
        
        parities = self.kernel(entries, padded_indices, valid_mask)
        
        expected = entries[0] ^ entries[1]
        self.assertTrue(torch.equal(parities[0], expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
