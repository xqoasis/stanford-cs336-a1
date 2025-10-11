import torch
from torch import nn
from cs336_basics import linear

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network
    
    SwiGLU is a variant of GLU (Gated Linear Unit) that uses SiLU activation.
    Formula: SwiGLU(x) = (W1(x) ⊙ SiLU(W3(x))) @ W2
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Three linear transformations
        self.w1 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)  # Gate projection
        self.w2 = linear.Linear(d_ff, d_model, device=device, dtype=dtype)  # Down projection
        self.w3 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)  # Up projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        # SwiGLU formula: (SiLU(W1(x)) ⊙ W3(x)) @ W2
        up = self.w1(x)  # (..., d_ff)  
        gate = self.w3(x)    # (..., d_ff)

        silu_up = up * torch.sigmoid(up)  # SiLU activation: x * sigmoid(x)
        gated = silu_up * gate  # Element-wise multiplication
        return self.w2(gated)   # (..., d_model)