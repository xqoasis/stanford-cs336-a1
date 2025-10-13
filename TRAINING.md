# Transformer Language Model Training Guide

Complete guide for training Transformer language models with the `train.py` script.

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Data Format](#data-format)
4. [Parameters](#parameters)
5. [Training Examples](#training-examples)
6. [Monitoring](#monitoring)
7. [Checkpoint Management](#checkpoint-management)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Implementation Details](#implementation-details)
11. [Testing](#testing)

---

## Features

The `train.py` training script implements the following features:

1. **Configurable Hyperparameters** - Control all model and training hyperparameters via command-line arguments
2. **Memory-Efficient Data Loading** - Load large datasets using `np.memmap` without loading entire dataset into memory
3. **Checkpoint Save/Resume** - Periodically save checkpoints and resume training from any checkpoint
4. **Training & Validation Monitoring** - Periodic evaluation on train and validation sets
5. **Flexible Logging** - Console logging and Weights & Biases integration
6. **Learning Rate Scheduling** - Cosine annealing with linear warmup
7. **Gradient Clipping** - Prevent gradient explosion

---

## Quick Start

### Basic Usage

Simplest training command:

```bash
uv run python train.py \
    --train_data data/train.npy \
    --vocab_size 10000 \
    --max_iters 1000
```

### Full Configuration Example

Training command with all important parameters:

```bash
uv run python train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --num_layers 6 \
    --learning_rate 3e-4 \
    --min_learning_rate 3e-5 \
    --weight_decay 0.1 \
    --batch_size 32 \
    --max_iters 10000 \
    --warmup_iters 1000 \
    --eval_interval 500 \
    --save_interval 1000 \
    --checkpoint_dir ./checkpoints \
    --device cuda \
    --seed 42
```

### Using Weights & Biases

Enable W&B logging:

```bash
uv run python train.py \
    --train_data data/train.npy \
    --vocab_size 10000 \
    --use_wandb \
    --wandb_project my-transformer \
    --wandb_run_name experiment-1
```

### Resume from Checkpoint

```bash
uv run python train.py \
    --train_data data/train.npy \
    --vocab_size 10000 \
    --resume_from checkpoints/checkpoint_iter_5000.pt \
    --max_iters 10000
```

---

## Data Format

The training script supports two data formats:

### 1. NumPy Array Files (.npy)

```python
import numpy as np

# Create or load tokenized data
tokens = np.array([1, 2, 3, 4, 5, ...], dtype=np.uint16)

# Save as .npy file
np.save('train.npy', tokens)
```

### 2. Raw Binary Files (.bin)

```python
import numpy as np

# Create or load tokenized data
tokens = np.array([1, 2, 3, 4, 5, ...], dtype=np.uint16)

# Save as raw binary file
tokens.tofile('train.bin')
```

---

## Parameters

### Data Arguments

- `--train_data`: Training data path (required)
- `--val_data`: Validation data path (optional)

### Model Hyperparameters (7 parameters)

- `--vocab_size`: Vocabulary size (required)
- `--context_length`: Context length/sequence length (default: 256)
- `--d_model`: Model dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8)
- `--d_ff`: Feed-forward network dimension (default: 2048)
- `--num_layers`: Number of Transformer layers (default: 6)
- `--eps`: Epsilon for RMSNorm (default: 1e-5)

### Optimizer Hyperparameters (6 parameters)

- `--learning_rate`: Maximum learning rate (default: 3e-4)
- `--min_learning_rate`: Minimum learning rate (default: 3e-5)
- `--weight_decay`: Weight decay coefficient (default: 0.1)
- `--beta1`: AdamW beta1 (default: 0.9)
- `--beta2`: AdamW beta2 (default: 0.95)
- `--grad_clip`: Maximum L2 norm for gradient clipping (default: 1.0)

### Training Arguments (4 parameters)

- `--batch_size`: Batch size (default: 32)
- `--max_iters`: Maximum training iterations (default: 10000)
- `--warmup_iters`: Learning rate warmup iterations (default: 1000)
- `--cosine_cycle_iters`: Cosine annealing cycle iterations (default: 10000)

### Logging and Saving Arguments (6 parameters)

- `--eval_interval`: Evaluation interval (default: 500)
- `--log_interval`: Logging interval (default: 10)
- `--save_interval`: Checkpoint saving interval (default: 1000)
- `--eval_iters`: Number of iterations for evaluation (default: 100)
- `--checkpoint_dir`: Checkpoint saving directory (default: ./checkpoints)
- `--resume_from`: Resume training from checkpoint

### System Arguments (6 parameters)

- `--device`: Training device, cuda or cpu (default: auto-detect)
- `--dtype`: Data type, float32/float16/bfloat16 (default: float32)
- `--seed`: Random seed (default: 42)
- `--use_wandb`: Use Weights & Biases logging
- `--wandb_project`: W&B project name
- `--wandb_run_name`: W&B run name

**Total**: 30+ configurable parameters

---

## Training Examples

### Scenario 1: Small Model Quick Experiment

```bash
uv run python train.py \
    --train_data data/train.npy \
    --vocab_size 5000 \
    --context_length 128 \
    --d_model 256 \
    --num_heads 4 \
    --d_ff 1024 \
    --num_layers 4 \
    --batch_size 64 \
    --max_iters 5000 \
    --log_interval 50 \
    --eval_interval 500
```

### Scenario 2: Medium Model Standard Training

```bash
uv run python train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --num_layers 6 \
    --batch_size 32 \
    --max_iters 50000 \
    --warmup_iters 2000 \
    --learning_rate 3e-4 \
    --checkpoint_dir ./checkpoints/medium-model \
    --use_wandb
```

### Scenario 3: Large Model Long Training

```bash
uv run python train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --vocab_size 50000 \
    --context_length 512 \
    --d_model 1024 \
    --num_heads 16 \
    --d_ff 4096 \
    --num_layers 12 \
    --batch_size 16 \
    --max_iters 100000 \
    --warmup_iters 5000 \
    --learning_rate 1e-4 \
    --min_learning_rate 1e-5 \
    --grad_clip 1.0 \
    --checkpoint_dir ./checkpoints/large-model \
    --save_interval 2000 \
    --device cuda \
    --dtype bfloat16
```

---

## Monitoring

### Console Output

During training, the script periodically outputs:

```
Iter 100/10000 | Loss: 4.5623 | LR: 3.00e-05 | Tokens/s: 12345
Iter 200/10000 | Loss: 4.2341 | LR: 6.00e-05 | Tokens/s: 12456
...

================================================================================
Evaluating...
Iter 500 | Train Loss: 3.8234 | Val Loss: 3.9123
================================================================================
```

### Weights & Biases

When using `--use_wandb`, training metrics are automatically logged to W&B, including:

- Training loss
- Validation loss
- Learning rate
- Throughput (tokens/s)

---

## Checkpoint Management

### Checkpoint File Structure

Checkpoint files contain:

```python
{
    'model': model.state_dict(),      # Model parameters
    'optimizer': optimizer.state_dict(),  # Optimizer state
    'iteration': iter_num             # Current iteration number
}
```

### Loading Checkpoint for Inference

```python
import torch
from cs336_basics.transformer import TransformerLM

# Initialize model
model = TransformerLM(
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6
)

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_final.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# Use model for inference
# ...
```

---

## Performance Optimization

### 1. Batch Size Tuning

- Increase `batch_size` when GPU memory is sufficient to improve throughput
- Decrease `batch_size` or use gradient accumulation when memory is insufficient

### 2. Mixed Precision Training

- Use `--dtype bfloat16` or `--dtype float16` to accelerate training
- bfloat16 is more stable, recommended for A100 and other supporting GPUs

### 3. Data Loading Optimization

- `.bin` format can achieve better loading performance
- Store data on local SSD rather than network storage

### 4. Learning Rate Tuning

- Larger models typically need smaller learning rates
- Increasing `warmup_iters` can improve training stability

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:
- Decrease `batch_size`
- Decrease `context_length`
- Decrease model size (`d_model`, `d_ff`, `num_layers`)
- Use mixed precision training (`--dtype bfloat16`)

### Issue: Loss is NaN

**Solutions**:
- Decrease learning rate (`--learning_rate`)
- Increase gradient clipping (`--grad_clip 0.5`)
- Increase warmup steps (`--warmup_iters`)
- Check data for anomalies

### Issue: Slow Training Speed

**Solutions**:
- Increase batch size
- Use mixed precision training
- Ensure data is on local storage
- Check GPU utilization

---

## Implementation Details

### Assignment Requirements

According to the assignment (Problem: training_together), the training script must implement:

1. **Configurable Hyperparameters** - Support all model and optimizer hyperparameters
2. **Memory-Efficient Data Loading** - Use `np.memmap` for large datasets
3. **Checkpoint Serialization** - Save to user-provided paths
4. **Periodic Logging** - Console output and Weights & Biases integration

### Core Training Loop

```python
# Data loading (using memmap)
def load_data_memmap(data_path, dtype=np.uint16):
    if data_path.endswith('.npy'):
        data = np.load(data_path, mmap_mode='r')
    elif data_path.endswith('.bin'):
        data = np.memmap(data_path, dtype=dtype, mode='r')
    return data

# Training loop
for iter_num in range(start_iter, args.max_iters):
    # Get learning rate
    lr = get_lr_cosine_schedule(...)
    
    # Get batch data
    inputs, targets = get_batch(train_data, ...)
    
    # Forward pass
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    
    # Backward pass + gradient clipping + optimization
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), args.grad_clip)
    optimizer.step()
    
    # Periodic evaluation and saving
    if (iter_num + 1) % args.eval_interval == 0:
        evaluate(...)
    if (iter_num + 1) % args.save_interval == 0:
        save_checkpoint(...)
```

### Key Features

#### 1. Memory Efficiency
- Use `np.memmap` to load data without loading entire dataset into memory
- Support `.npy` and `.bin` data formats
- Support mixed precision training (float16/bfloat16)

#### 2. Recoverability
- Automatically save checkpoints (containing model, optimizer, and iteration)
- Can resume training from any checkpoint
- Checkpoint filenames include iteration numbers for easy management

#### 3. Monitoring and Logging
- Real-time console output (loss, learning rate, throughput)
- Periodic evaluation on train and validation sets
- Weights & Biases integration (optional)
- Training logs automatically saved to file

#### 4. Flexibility
- All hyperparameters configurable via command-line
- Support multiple devices (CPU/CUDA)
- Support multiple data types
- Easy to extend and modify

### Correspondence with Assignment Requirements

| Requirement | Implementation | File/Code |
|-------------|----------------|-----------|
| Configurable hyperparameters | | `parse_args()` in `train.py`, 30+ parameters |
| Memory-efficient loading | | `load_data_memmap()` in `train.py` |
| Checkpoint serialization | | `save_checkpoint()` and `load_checkpoint()` in `training_utils.py` |
| Periodic logging | | Main loop in `train.py`, console + W&B support |

---

## Testing

### Run Test Script

Verify all functionality with the test script:

```bash
uv run python test_train_script.py
```

### Test Coverage

The test script validates:

1. Module imports
2. Data creation and loading
3. Model initialization
4. Optimizer setup
5. Batch generation
6. Forward pass
7. Training step (including backward pass and gradient clipping)
8. Learning rate schedule
9. Checkpoint save and load
10. Script file verification

---

## Complete Example Script

Create a complete training script `run_training.sh`:

```bash
#!/bin/bash

# Set parameters
TRAIN_DATA="data/train.npy"
VAL_DATA="data/val.npy"
CHECKPOINT_DIR="./checkpoints/$(date +%Y%m%d_%H%M%S)"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Run training
uv run python train.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --num_layers 6 \
    --learning_rate 3e-4 \
    --min_learning_rate 3e-5 \
    --weight_decay 0.1 \
    --batch_size 32 \
    --max_iters 50000 \
    --warmup_iters 2000 \
    --eval_interval 500 \
    --log_interval 10 \
    --save_interval 1000 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --device cuda \
    --seed 42 \
    --use_wandb \
    --wandb_project transformer-lm \
    --wandb_run_name "experiment-$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee $CHECKPOINT_DIR/training.log

echo "Training completed! Checkpoints saved in: $CHECKPOINT_DIR"
```

Usage:

```bash
chmod +x run_training.sh
./run_training.sh
```

---

## Related Files

- `train.py`: Main training script
- `cs336_basics/transformer.py`: Model implementation
- `cs336_basics/training_utils.py`: Training utilities
- `cs336_basics/optimizer.py`: Optimizer implementation
- `test_train_script.py`: Test script

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)

---