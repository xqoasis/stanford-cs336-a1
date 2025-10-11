import torch
from torch import nn
from cs336_basics import embedding, linear, rope, rmsnorm, mha, swiglu

class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block
    
    Architecture:
    1. Layer norm -> Multi-head self-attention -> Residual connection
    2. Layer norm -> Feed-forward network -> Residual connection
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, eps: float = 1e-5, 
                 max_seq_len: int = 2048, theta: float = 10000.0, use_rope: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Layer normalization (RMSNorm)
        self.ln1 = rmsnorm.RMSNorm(d_model, eps, device=device, dtype=dtype)  # Before attention
        self.ln2 = rmsnorm.RMSNorm(d_model, eps, device=device, dtype=dtype)  # Before FFN
        
        # Multi-head self-attention with RoPE support
        self.attn = mha.MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len=max_seq_len, 
            theta=theta, use_rope=use_rope, device=device, dtype=dtype
        )
        
        # Feed-forward network (SwiGLU)
        self.ffn = swiglu.SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply pre-norm transformer block
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-norm attention sublayer
        # x -> LayerNorm -> Attention -> Residual
        norm_x = self.ln1(x)
        attn_output = self.attn(norm_x, mask)
        x = x + attn_output  # Residual connection
        
        # Pre-norm feed-forward sublayer  
        # x -> LayerNorm -> FFN -> Residual
        norm_x = self.ln2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output   # Residual connection
        
        return x


class TransformerLM(nn.Module):
    """Transformer Language Model
    
    A complete autoregressive language model following the GPT architecture:
    1. Token embeddings
    2. Multiple Transformer blocks
    3. Final layer norm
    4. Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        eps: float = 1e-5,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embedding = embedding.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, eps, 
                max_seq_len=context_length, theta=10000.0, use_rope=True,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_final = rmsnorm.RMSNorm(d_model, eps, device=device, dtype=dtype)
        
        # Output projection to vocabulary (language modeling head)
        self.lm_head = linear.Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Transformer Language Model
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len)
            
        Returns:
            Logits over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)  # Each block applies causal masking
        
        # Final layer norm
        x = self.ln_final(x)  # (batch_size, seq_len, d_model)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits

