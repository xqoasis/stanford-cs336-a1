import torch
from torch import nn

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