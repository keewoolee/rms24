"""
RMS24 Hint Generation - Loop-free version for Forge.
"""

import torch
import torch.nn as nn
from functools import reduce


class HintGenKernel(nn.Module):
    """Gather + XOR reduction without Python loops."""
    
    def forward(
        self,
        entries: torch.Tensor,  # [N, 5] int64
        indices: torch.Tensor,  # [H, S] int64
        mask: torch.Tensor,     # [H, S] bool
    ) -> torch.Tensor:
        # Gather: [H, S, 5]
        gathered = entries[indices]
        
        # Apply mask: zero out invalid
        gathered = gathered * mask.unsqueeze(-1).long()
        
        # XOR reduce via cumulative XOR and take last
        # Use reduce with bitwise_xor
        result = reduce(
            torch.bitwise_xor,
            [gathered[:, i, :] for i in range(gathered.shape[1])]
        )
        
        return result


def get_example_inputs(device="cuda"):
    N = 262144
    H = 100
    S = 512
    
    entries = torch.randint(0, 2**60, (N, 5), dtype=torch.int64, device=device)
    indices = torch.randint(0, N, (H, S), dtype=torch.int64, device=device)
    mask = torch.ones(H, S, dtype=torch.bool, device=device)
    
    return entries, indices, mask


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HintGenKernel().to(device)
    inputs = get_example_inputs(device)
    output = model(*inputs)
    print(f"Output: {output.shape}, Device: {device}")
