import torch
import numpy as np
import math


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss between logits and targets with numerical stability.
    Cross-entropy loss formula: ℓ_i = -log(softmax(o_i)[x_{i+1}])

    Args:
        logits: Unnormalized logits of shape (..., vocab_size)
        targets: Target class indices of shape (...)
        
    Returns:
        Average cross-entropy loss across all examples (scalar)
    """
    # Get the shape information
    *batch_dims, vocab_size = logits.shape
    batch_size = logits.numel() // vocab_size
    # Flatten to (batch_size, vocab_size) for easier processing
    logits_flat = logits.view(batch_size, vocab_size)
    targets_flat = targets.view(batch_size)

    # step 1: substract max for numerical stability
    # max_logits shape: (batch_size, 1)
    max_logits = torch.max(logits_flat, dim=1, keepdim=True)[0]
    logits_stable = logits_flat - max_logits

    # step 2: compute log softmax
    # log_softmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))
    exp_logits = torch.exp(logits_stable)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)

    # log_softmax = logits_stable - log_sum_exp
    log_softmax = logits_stable - log_sum_exp

    # step 3: gather the log probabilities of the targets
    # use advanced indexing to get log_softmax[i, targets[i]] for each i
    target_log_probs = log_softmax[torch.arange(batch_size), targets_flat]

    # step 4: compute negative log likelihood loss
    losses = -target_log_probs

    return torch.mean(losses)


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    Clip gradients to have L2 norm at most max_l2_norm.

    This prevents gradient explosion by scaling down gradients when their
    combined L2 norm exceeds the threshold.

    Args:
        parameters: Iterable of parameters with .grad attributes
        max_l2_norm: Max allowed L2 norm of gradients
    """
    # Collect all gradients
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad.view(-1))  # Flatten each gradient
    
    if not gradients:
        return  # No gradients to clip
    
    # Concatenate all gradients into a single vector
    all_gradients = torch.cat(gradients)

    # Compute L2 norm of all gradients combined
    # global l2 norm: sqrt(sum(g_i^2))
    total_norm = torch.norm(all_gradients, p=2)

    # Compute clipping factor (i.e. scaling factor)
    clip_factor = max_l2_norm / (total_norm + 1e-8)  # Add small epsilon to avoid division by zero

    # only clip if the norm exceeds the threshold
    if clip_factor < 1.0:
        # Scale down all gradients by the same factor
        # only change the gradient's magnitude, without changing the direction
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_factor)


def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    The schedule has three phases:
    1. Linear warmup: [0, warmup_iters) - linearly increase from 0 to max_lr
    2. Cosine annealing: [warmup_iters, cosine_cycle_iters) - 
       cosine decay from max_lr to min_lr
    3. Constant: [cosine_cycle_iters, inf) - stay at min_lr
    
    Args:
        it: Current iteration number
        max_learning_rate: α_max, maximum learning rate
        min_learning_rate: α_min, minimum learning rate
        warmup_iters: T_w, number of warmup iterations
        cosine_cycle_iters: T_c, number of cosine annealing iterations
        
    Returns:
        Learning rate at iteration it
    """
    # p1: linear warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_factor
    else:
        return min_learning_rate


def get_batch(dataset, batch_size: int, context_length: int, device: str):
    """
    Sample language modeling input sequences and their corresponding labels from the dataset.
    
    For language modeling, the labels are simply the input shifted by one position.
    For example, if input is [1, 2, 3, 4], the label is [2, 3, 4, 5].
    
    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0')
        
    Returns:
        Tuple of (inputs, labels), both of shape (batch_size, context_length)
    """
    # 随机生成起始点：
    # 创建一个包含 batch_size (B) 条数据的批次。
    # 随机生成 B 个整数，作为每一条训练序列在那个巨大数组中的起始索引。
    # 为了确保能切分出完整的序列，这些随机索引的范围应该在 0 到 (数组总长度 - context_length) 之间。
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    inputs = []
    labels = []

    # 切分输入 x 和目标 y：
    # 遍历刚刚生成的 B 个起始索引。对于每一个起始索引 i：
    # 输入 x：就是从 i 开始，长度为 context_length (L) 的序列。tokenized_data[i : i + L]。
    # 目标 y：就是 x 整体向右移动一位的序列. tokenized_data[i+1 : i + L + 1]。
    for start_idx in start_indices:
        seq_input = dataset[start_idx : start_idx + context_length]
        seq_label = dataset[start_idx + 1 : start_idx + context_length + 1]
        # 堆叠成批次 (Stack)
        inputs.append(seq_input)
        labels.append(seq_label)

    # 转换为 PyTorch 张量：
    # 两个 Numpy 数组转换为 PyTorch Tensors，并把它们放到正确的设备上（比如 'cuda' 或 'cpu'）。
    inputs = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    labels = torch.tensor(np.array(labels), dtype=torch.long, device=device)
    
    return inputs, labels


def save_checkpoint(model, optimizer, iteration: int, out):
    """
    Saves the model, optimizer, and iteration number to a checkpoint file.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer to save
        iteration: Current iteration number
        out: Output path for the checkpoint file
    """
    # Create a dictionary to hold all the state we need to save.
    checkpoint = {
        'model': model.state_dict(),      # Get model's state dictionary
        'optimizer': optimizer.state_dict(), # Get optimizer's state dictionary
        'iteration': iteration,             # Save the current iteration number
    }
    # Use torch.save to dump the dictionary to a file.
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """
    Loads a checkpoint and restores the model and optimizer states.
    Returns the iteration number to resume training from.
    
    Args:
        src: Path to the checkpoint file
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into
        
    Returns:
        int: Iteration number to resume training from
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']
    return iteration

