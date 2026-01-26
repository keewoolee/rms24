"""
RMS24 Hint Generation - Minimal kernel for Forge optimization.

Operation: Gather entries by index, XOR reduce to parity.
Target: H200 GPU, CUDA output
"""

import torch
import torch.nn as nn


class HintGenKernel(nn.Module):
    """
    Batched gather + XOR reduction.
    
    Forward args:
        entries: [N, 5] int64 - database
        indices: [H, S] int64 - gather indices per hint
        mask: [H, S] bool - valid mask
    
    Returns:
        [H, 5] int64 - XOR parity per hint
    """
    
    def forward(
        self,
        entries: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        gathered = entries[indices]
        gathered = gathered * mask.unsqueeze(-1).long()
        
        parity = gathered[:, 0, :]
        for i in range(1, gathered.shape[1]):
            parity = parity ^ gathered[:, i, :]
        
        return parity


def get_example_inputs(device="cuda"):
    """Example inputs for benchmarking."""
    N = 262144  # num entries
    H = 100     # num hints
    S = 512     # subset size
    
    entries = torch.randint(0, 2**60, (N, 5), dtype=torch.int64, device=device)
    indices = torch.randint(0, N, (H, S), dtype=torch.int64, device=device)
    mask = torch.ones(H, S, dtype=torch.bool, device=device)
    
    return entries, indices, mask


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HintGenKernel().to(device)
    inputs = get_example_inputs(device)
    
    output = model(*inputs)
    print(f"Output shape: {output.shape}")
    print(f"Device: {device}")
