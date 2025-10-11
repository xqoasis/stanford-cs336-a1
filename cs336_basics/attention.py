import torch
from torch import Tensor
from jaxtyping import Float


def softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
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
    attention_weights = softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, V)
    
    return output

