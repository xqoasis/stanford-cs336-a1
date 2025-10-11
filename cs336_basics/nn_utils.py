import torch
from torch import nn
from cs336_basics import linear, embedding, rmsnorm, swiglu, rope, mha, transformer, attention
from cs336_basics.training_utils import (
    cross_entropy_loss,
    gradient_clipping,
    get_lr_cosine_schedule,
    get_batch,
    save_checkpoint,
    load_checkpoint
)
from jaxtyping import Float, Int
from torch import Tensor

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
    linear_layer = linear.Linear(d_in, d_out)
    
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
    embedding_layer = embedding.Embedding(vocab_size, d_model) # the matrix is (vocab_size, d_model)
    
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
    rmsnorm_layer = rmsnorm.RMSNorm(d_model, eps)
    
    # Load the provided weights using state_dict
    state_dict = {'weight': weights}
    rmsnorm_layer.load_state_dict(state_dict)
    
    # Run forward pass
    return rmsnorm_layer(in_features)

def run_silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    Apply SiLU (Swish) activation function to the input features.
    
    SiLU(x) = x * sigmoid(x)

    Args:
        in_features (Float[Tensor, "..."]): Input features to run SiLU on
    
    Returns:
        Float[Tensor, "..."]: Tensor with SiLU applied element-wise
    """
    return in_features * torch.sigmoid(in_features)
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
    swiglu_layer = swiglu.SwiGLU(d_model, d_ff)
    
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
    ropeOutput = rope.RotaryPositionEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    # Apply RoPE
    return ropeOutput(in_query_or_key, token_positions)

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
    return attention.scaled_dot_product_attention(Q, K, V, mask)


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
    mha_layer = mha.MultiHeadSelfAttention(d_model, num_heads, use_rope=False)
    
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
    mha_layer = mha.MultiHeadSelfAttention(
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
    block = transformer.TransformerBlock(
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
    model = transformer.TransformerLM(
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


# Training utilities moved to cs336_basics/training_utils.py
# - cross_entropy_loss
# - gradient_clipping  
# - get_lr_cosine_schedule
# - get_batch
# - save_checkpoint
# - load_checkpoint