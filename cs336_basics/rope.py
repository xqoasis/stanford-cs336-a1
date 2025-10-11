import torch
from torch import nn
class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) implementation
    
    RoPE rotates query and key vectors by position-dependent angles to inject
    positional information into the attention mechanism.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: Θ value for the RoPE
            d_k: dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be inputted
            device: Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Pre-compute frequency values for efficient computation
        # freq_k = theta^(-2k/d_k) for k in {0, 1, ..., d_k/2 - 1}
        k_values = torch.arange(0, d_k // 2, dtype=torch.float, device=device)
        freqs = theta ** (-2 * k_values / d_k ) # (d_k/2,)

        # Create position tensor for all possible positions
        positions = torch.arange(max_seq_len, dtype=torch.float, device=device) #(max_seq_len,)

        # Compute all possible angles:position * freq
        # angles[i, k] = position_i * freq_k
        angles = torch.outer(positions, freqs) #(max_seq_len, d_k/2)

        # Pre-compute cos and sin values
        cos_vals = torch.cos(angles) #(max_seq_len, d_k/2)
        sin_vals = torch.sin(angles) #(max_seq_len, d_k/2)
        
        # Create buffers for position indices and rotation angles
        self.register_buffer('cos_vals', cos_vals, persistent=False)
        self.register_buffer('sin_vals', sin_vals, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        Process an input tensor and apply RoPE.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)
            
        Returns:
            Tensor of the same shape as x with RoPE applied
        """
        # Get the shape info
        *batch_dims, seq_len, d_k = x.shape
        
        # Reshape x to separate even and odd dimensions
        # x: (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2)
        x_reshaped = x.view(*batch_dims, seq_len, d_k // 2, 2)
        
        # Extract even and odd elements
        x_even = x_reshaped[..., 0]  # (..., seq_len, d_k/2) - positions 0, 2, 4, ...
        x_odd = x_reshaped[..., 1]   # (..., seq_len, d_k/2) - positions 1, 3, 5, ...
        
        # Get cos and sin values for the token positions (descrete positions to interpolate)
        # token_positions: (..., seq_len)
        cos = self.cos_vals[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_vals[token_positions]  # (..., seq_len, d_k/2)
        
        # Apply rotation: 
        # [x_even'] = [cos  -sin] [x_even]
        # [x_odd' ]   [sin   cos] [x_odd ]
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos
        
        # Combine back
        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)  # (..., seq_len, d_k/2, 2)
        
        # Reshape back to original shape
        x_rotated = x_rotated.view(*batch_dims, seq_len, d_k)
        
        return x_rotated     