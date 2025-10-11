import torch
from torch import nn
from cs336_basics import linear, rope, attention

class MultiHeadSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention implementation with RoPE support"""
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, 
                 theta: float = 10000.0, use_rope: bool = True, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        
        # Linear projections for Q, K, V (all heads combined)
        self.q_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Output projection
        self.output_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        
        # RoPE for positional encoding
        if use_rope:
            self.rope = rope.RotaryPositionEmbedding(
                theta=theta, 
                d_k=self.d_k, 
                max_seq_len=max_seq_len, 
                device=device
            )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, 
                token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Apply causal multi-head self-attention with optional RoPE
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len)
                 If None, a causal mask will be automatically created
            token_positions: Optional token positions for RoPE, shape (batch_size, seq_len)
                           If None and RoPE is enabled, will use range(seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Generate causal mask if none provided
        if mask is None:
            # Create causal mask: lower triangular matrix
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # Handle mask dimensions
        if mask.dim() == 2:  # (seq_len, seq_len)
            mask = mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        
        # Generate token positions for RoPE if needed
        if self.use_rope and token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, d_model)
        V = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Reshape and transpose for multi-head attention
        # Split d_model into num_heads * d_k
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        
        # Apply RoPE to Q and K (not V)
        if self.use_rope:
            # Expand token_positions for multi-head
            rope_positions = token_positions.unsqueeze(1).expand(batch_size, self.num_heads, seq_len)
            Q = self.rope(Q, rope_positions)
            K = self.rope(K, rope_positions)
        
        # Expand mask for multi-head attention
        # mask: (batch_size, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        mask = mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
        
        # Apply scaled dot-product attention
        attention_output = attention.scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_len, d_k)
        
        # Reshape back to concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Apply output projection
        output = self.output_proj(attention_output)
        
        return output

