import torch
from torch import nn

class Linear(nn.Module):
    """Linear layer implementation from scratch (without bias)
    
    Matches PyTorch's nn.Linear interface except for the absence of bias.
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create weight parameter W (not W^T) for memory ordering reasons
        # Shape: (out_features, in_features) to match PyTorch's nn.Linear
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        self.reset_parameters()
            
    def reset_parameters(self):
        """Initialize parameters using truncated normal distribution"""
        torch.nn.init.trunc_normal_(self.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the linear layer
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Linear transformation: x @ W^T (where W is stored as (out_features, in_features))
        return torch.nn.functional.linear(x, self.weight, bias=None)