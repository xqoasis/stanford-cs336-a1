import torch
from torch import nn
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization implementation
    
    RMSNorm normalizes the input using root mean square instead of variance.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # Learnable scale parameter
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of the same shape
        """
        # Compute RMS along the last dimension
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return x / rms * self.weight