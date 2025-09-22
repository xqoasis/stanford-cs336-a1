import torch
from torch import nn
from jaxtyping import Float, Int
from torch import Tensor

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

class Embedding(nn.Module):
    """Embedding layer implementation from scratch
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        - num_embeddings (int): Size of the vocabulary
        - embedding_dim (int): Dimension of the embedding vectors
        - device (torch.device, optional): Device to store the parameters on
        - dtype (torch.dtype, optional): Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding weight matrix with shape (num_embeddings, embedding_dim)
        # This stores d_model as the final dimension as required
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using truncated normal distribution"""
        torch.nn.init.trunc_normal_(self.weight)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the embeddings for the given token IDs
        
        Args:
            token_ids: Token IDs tensor of shape (batch_size, sequence_length) or any shape
            
        Returns:
            Embedding vectors of shape (*token_ids.shape, embedding_dim)
        """
        # Use advanced indexing to lookup embeddings
        # token_ids should contain indices into the vocabulary
        return self.weight[token_ids]


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
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Gate projection
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # Down projection
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Up projection
    
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

def scaled_dot_product_attention(
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
    """
    Scaled Dot-Product Attention implementation
    
    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
        
    Returns:
        Attention output of shape (..., seq_len_q, d_v)
    """
    # Get the dimensionality for scaling
    d_k = Q.size(-1)

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # torch.tensor(d_k, dtype=Q.dtype, device=Q.device) is to ensure the dtype and device are the same as Q
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    
    # Apply mask if provided (set masked positions to large negative value)
    # mask before softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # Apply softmax to get attention weights
    attention_weights = run_softmax(scores, dim = -1)
    # Apply attention weights to values
    output = torch.matmul(attention_weights, V)
    
    return output


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
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Output projection
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # RoPE for positional encoding
        if use_rope:
            self.rope = RotaryPositionEmbedding(
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
        attention_output = scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_len, d_k)
        
        # Reshape back to concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Apply output projection
        output = self.output_proj(attention_output)
        
        return output


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
        self.ln1 = RMSNorm(d_model, eps, device=device, dtype=dtype)  # Before attention
        self.ln2 = RMSNorm(d_model, eps, device=device, dtype=dtype)  # Before FFN
        
        # Multi-head self-attention with RoPE support
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len=max_seq_len, 
            theta=theta, use_rope=use_rope, device=device, dtype=dtype
        )
        
        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
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
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
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
        self.ln_final = RMSNorm(d_model, eps, device=device, dtype=dtype)
        
        # Output projection to vocabulary (language modeling head)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
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


def run_linear(d_in, d_out, weights, in_features) -> Float[Tensor, "..."]:
    """
    Run a linear layer with the given weights and input features.

    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension
        weights (Float[Tensor, "d_out d_in"]): Weight matrix
        in_features (Float[Tensor, "..."]): Input features
    """
    # Create a linear layer
    linear_layer = Linear(d_in, d_out)
    
    # Load the provided weights using state_dict
    state_dict = {'weight': weights}
    linear_layer.load_state_dict(state_dict)
    
    # Run forward pass
    return linear_layer(in_features)


def run_embedding(vocab_size: int, d_model: int, weights: Float[Tensor, "vocab_size d_model"], token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
    """
    Run an embedding layer with the given weights and token IDs.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by the Embedding layer.
    """
    # Create an embedding layer
    embedding_layer = Embedding(vocab_size, d_model) # the matrix is (vocab_size, d_model)
    
    # Load the provided weights using state_dict
    state_dict = {'weight': weights}
    embedding_layer.load_state_dict(state_dict)
    
    # Run forward pass
    return embedding_layer(token_ids)


def run_rmsnorm(d_model: int, eps: float, weights: Float[Tensor, "d_model"], in_features: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    """
    Run RMSNorm with the given weights and input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input
        eps (float): A value added to the denominator for numerical stability
        weights (Float[Tensor, "d_model"]): RMSNorm weights
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on
    
    Returns:
        Float[Tensor, "... d_model"]: Tensor with the output of running RMSNorm
    """
    # Create RMSNorm layer
    rmsnorm_layer = RMSNorm(d_model, eps)
    
    # Load the provided weights using state_dict
    state_dict = {'weight': weights}
    rmsnorm_layer.load_state_dict(state_dict)
    
    # Run forward pass
    return rmsnorm_layer(in_features)


def run_silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    Apply SiLU (Swish) activation function to the input features.

    Args:
        in_features (Float[Tensor, "..."]): Input features to run SiLU on
    
    Returns:
        Float[Tensor, "..."]: Tensor with SiLU applied element-wise
    """
    return torch.nn.functional.silu(in_features)


def run_swiglu(d_model: int, d_ff: int, w1_weight: Float[Tensor, "d_ff d_model"], w2_weight: Float[Tensor, "d_model d_ff"], w3_weight: Float[Tensor, "d_ff d_model"], in_features: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    """
    Run SwiGLU with the given weights and input features.

    Args:
        d_model (int): Dimensionality of the feedforward input and output
        d_ff (int): Dimensionality of the up-project happening internally
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer
    
    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as input
    """
    # Create SwiGLU layer
    swiglu_layer = SwiGLU(d_model, d_ff)
    
    # Load the provided weights using state_dict
    state_dict = {
        'w1.weight': w1_weight,
        'w2.weight': w2_weight,
        'w3.weight': w3_weight
    }
    swiglu_layer.load_state_dict(state_dict)
    
    # Run forward pass
    return swiglu_layer(in_features)

def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, "... sequence_length d_k"],
        token_positions: Int[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length d_k"]:
    """
    Run RoPE with the given parameters.

    Args:
        d_k: Embedding dimension size for the query or key tensor
        theta: RoPE parameter
        max_seq_len: Maximum sequence length to pre-cache
        in_query_or_key: Input tensor to run RoPE on
        token_positions: Tensor with the token positions
        
    Returns:
        Tensor with RoPE applied
    """
    rope = RotaryPositionEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    # Apply RoPE
    return rope(in_query_or_key, token_positions)

def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Apply softmax to the specified dimension of the input tensor.
    
    Uses the numerical stability trick of subtracting the maximum value
    to avoid overflow in the exponential function.
    
    Softmax formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax
        dim (int): Dimension to apply softmax to
    
    Returns:
        Float[Tensor, "..."]: Tensor with softmax applied to the specified dimension
    """
    # Step 1: Find the maximum value along the specified dimension
    # keepdim=True maintains the dimension for broadcasting
    max_vals = torch.max(in_features, dim=dim, keepdim=True)[0]
    
    # Step 2: Subtract max for numerical stability
    # This prevents overflow when computing exp(large_number)
    shifted = in_features - max_vals
    
    # Step 3: Compute exponentials
    exp_vals = torch.exp(shifted)
    
    # Step 4: Compute sum of exponentials along the specified dimension
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)
    
    # Step 5: Normalize to get probabilities
    softmax_output = exp_vals / sum_exp
    
    return softmax_output


def run_scaled_dot_product_attention(
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        mask: Float[Tensor, "... queries keys"] = None,
        device=None,
        dtype=None,
    ) -> Float[Tensor, "... queries d_v"]:
    """
    Run scaled dot-product attention with the given Q, K, V tensors.

    Args:
        Q: Query tensor
        K: Key tensor  
        V: Value tensor
        mask: Optional attention mask
    
    Returns:
        Attention output tensor
    """
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, "d_k d_in"],
        k_proj_weight: Float[Tensor, "d_k d_in"], 
        v_proj_weight: Float[Tensor, "d_v d_in"],
        o_proj_weight: Float[Tensor, "d_model d_v"],
        in_features: Float[Tensor, "... sequence_length d_in"],
    ) -> Float[Tensor, "... sequence_length d_out"]:
    """
    Run multi-head self-attention with the given weights and input features.

    Args:
        d_model: Model dimensionality
        num_heads: Number of attention heads
        q_proj_weight: Query projection weights
        k_proj_weight: Key projection weights  
        v_proj_weight: Value projection weights
        o_proj_weight: Output projection weights
        in_features: Input features
    
    Returns:
        Multi-head attention output
    """
    # Create multi-head attention layer (without RoPE for this adapter)
    mha_layer = MultiHeadSelfAttention(d_model, num_heads, use_rope=False)
    
    # Load the provided weights using state_dict
    state_dict = {
        'q_proj.weight': q_proj_weight,
        'k_proj.weight': k_proj_weight,
        'v_proj.weight': v_proj_weight,
        'output_proj.weight': o_proj_weight
    }
    mha_layer.load_state_dict(state_dict)
    
    # Run forward pass (with causal masking automatically applied)
    return mha_layer(in_features)


def run_multihead_self_attention_with_rope(
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, "d_k d_in"],
        k_proj_weight: Float[Tensor, "d_k d_in"],
        v_proj_weight: Float[Tensor, "d_v d_in"],
        o_proj_weight: Float[Tensor, "d_model d_v"],
        in_features: Float[Tensor, "... sequence_length d_in"],
        token_positions: Int[Tensor, "... sequence_length"] = None,
    ) -> Float[Tensor, "... sequence_length d_out"]:
    """
    Run multi-head self-attention with RoPE and the given weights and input features.

    Args:
        d_model: Model dimensionality
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for RoPE
        theta: RoPE theta parameter
        q_proj_weight: Query projection weights
        k_proj_weight: Key projection weights  
        v_proj_weight: Value projection weights
        o_proj_weight: Output projection weights
        in_features: Input features
        token_positions: Optional token positions for RoPE
    
    Returns:
        Multi-head attention output with RoPE applied
    """
    # Create multi-head attention layer with RoPE
    mha_layer = MultiHeadSelfAttention(
        d_model=d_model, 
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        use_rope=True
    )
    
    # Load the provided weights using state_dict
    state_dict = {
        'q_proj.weight': q_proj_weight,
        'k_proj.weight': k_proj_weight,
        'v_proj.weight': v_proj_weight,
        'output_proj.weight': o_proj_weight
    }
    mha_layer.load_state_dict(state_dict)
    
    # Run forward pass with RoPE
    return mha_layer(in_features, token_positions=token_positions)


def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, "... sequence_length d_k"],
        token_positions: Int[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.
    
    Args:
        d_k: Embedding dimension size for the query or key tensor
        theta: RoPE parameter
        max_seq_len: Maximum sequence length to pre-cache
        in_query_or_key: Input tensor to run RoPE on
        token_positions: Tensor with the token positions
        
    Returns:
        Tensor with RoPE applied
    """
    # Create RoPE module
    rope = RotaryPositionEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    
    # Apply RoPE
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, torch.Tensor],
        in_features: Float[Tensor, "batch sequence_length d_model"],
    ) -> Float[Tensor, "batch sequence_length d_model"]:
    """
    Run a pre-norm transformer block with the given weights and input features.

    Args:
        d_model: Model dimensionality
        num_heads: Number of attention heads
        d_ff: Feed-forward network inner dimension
        max_seq_len: Maximum sequence length (for RoPE, if implemented)
        theta: RoPE theta parameter (for RoPE, if implemented)
        weights: Dictionary containing all the weights for the transformer block
        in_features: Input features
    
    Returns:
        Transformer block output
    """
    # Create transformer block with RoPE
    block = TransformerBlock(
        d_model, num_heads, d_ff, 
        max_seq_len=max_seq_len, theta=theta, use_rope=True
    )
    
    # Load the provided weights using state_dict
    # Note: We need to match the expected weight keys from the test
    state_dict = {
        'ln1.weight': weights['ln1.weight'],
        'ln2.weight': weights['ln2.weight'],
        'attn.q_proj.weight': weights['attn.q_proj.weight'],
        'attn.k_proj.weight': weights['attn.k_proj.weight'], 
        'attn.v_proj.weight': weights['attn.v_proj.weight'],
        'attn.output_proj.weight': weights['attn.output_proj.weight'],
        'ffn.w1.weight': weights['ffn.w1.weight'],
        'ffn.w2.weight': weights['ffn.w2.weight'],
        'ffn.w3.weight': weights['ffn.w3.weight'],
    }
    block.load_state_dict(state_dict)
    
    # Run forward pass
    return block(in_features)


def run_transformer_lm(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        weights: dict[str, torch.Tensor],
        input_ids: Int[Tensor, "batch sequence_length"],
    ) -> Float[Tensor, "batch sequence_length vocab_size"]:
    """
    Run a transformer language model with the given weights and input ids.

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length
        d_model: Model dimensionality
        num_heads: Number of attention heads
        d_ff: Feed-forward network inner dimension
        num_layers: Number of transformer blocks
        weights: Dictionary containing all the weights for the transformer LM
        input_ids: Input token IDs
    
    Returns:
        Logits over vocabulary
    """
    # Create transformer language model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    
    # Convert weight keys from test format to our model format
    converted_weights = {}
    for key, value in weights.items():
        if key == 'token_embeddings.weight':
            converted_weights['token_embedding.weight'] = value
        elif key == 'lm_head.weight':
            converted_weights['lm_head.weight'] = value  
        elif key == 'ln_final.weight':
            converted_weights['ln_final.weight'] = value
        elif key.startswith('layers.'):
            # Convert layers.X.* to blocks.X.*
            new_key = key.replace('layers.', 'blocks.')
            converted_weights[new_key] = value
        else:
            converted_weights[key] = value
    
    # Load the converted weights using state_dict
    model.load_state_dict(converted_weights)
    
    # Run forward pass
    return model(input_ids)
