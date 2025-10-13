#!/usr/bin/env python3
"""
Training script for Transformer language model

Features:
- Configurable model and optimizer hyperparameters
- Memory-efficient loading of large datasets using np.memmap
- Checkpoint saving and resuming
- Periodic logging of training and validation performance
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Import custom modules
from cs336_basics.transformer import TransformerLM
from cs336_basics.training_utils import (
    cross_entropy_loss,
    gradient_clipping,
    get_lr_cosine_schedule,
    get_batch,
    save_checkpoint,
    load_checkpoint
)
from cs336_basics.optimizer import AdamW


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Transformer language model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Training data path (.npy or .bin file)')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Validation data path (.npy or .bin file)')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, required=True,
                        help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256,
                        help='Context length (sequence length)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward network dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of Transformer layers')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='Epsilon for RMSNorm')
    
    # Optimizer hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Maximum learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=3e-5,
                        help='Minimum learning rate (after cosine annealing)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay coefficient')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.95,
                        help='AdamW beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_iters', type=int, default=10000,
                        help='Maximum training iterations')
    parser.add_argument('--warmup_iters', type=int, default=1000,
                        help='Learning rate warmup iterations')
    parser.add_argument('--cosine_cycle_iters', type=int, default=10000,
                        help='Cosine annealing cycle iterations')
    
    # Logging and saving arguments
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluation interval (iterations)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (iterations)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Checkpoint saving interval (iterations)')
    parser.add_argument('--eval_iters', type=int, default=100,
                        help='Number of iterations for evaluation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint saving directory')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device (cuda or cpu)')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Data type')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='transformer-lm',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name')
    
    return parser.parse_args()


def load_data_memmap(data_path, dtype=np.uint16):
    """Load large datasets efficiently using memmap"""
    if data_path.endswith('.npy'):
        # For .npy files, use numpy's memmap
        data = np.load(data_path, mmap_mode='r')
    elif data_path.endswith('.bin'):
        # For raw binary files
        data = np.memmap(data_path, dtype=dtype, mode='r')
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    return data


def estimate_loss(model, train_data, val_data, eval_iters, batch_size, context_length, device):
    """Estimate loss on training and validation sets"""
    model.eval()
    losses = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        if data is None:
            continue
            
        split_losses = []
        for _ in range(eval_iters):
            # Get batch data
            inputs, targets = get_batch(data, batch_size, context_length, device)
            
            # Forward pass
            with torch.no_grad():
                logits = model(inputs)
                loss = cross_entropy_loss(logits, targets)
                split_losses.append(loss.item())
        
        losses[split] = np.mean(split_losses)
    
    model.train()
    return losses


def get_dtype(dtype_str):
    """Convert string to torch dtype"""
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.float32)


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
        except ImportError:
            print("Warning: Unable to import wandb, W&B logging disabled")
            args.use_wandb = False
    
    # Set device and data type
    device = args.device
    dtype = get_dtype(args.dtype)
    print(f"Using device: {device}, data type: {dtype}")
    
    # Load data (using memmap to save memory)
    print(f"Loading training data: {args.train_data}")
    train_data = load_data_memmap(args.train_data)
    print(f"Training data size: {len(train_data):,} tokens")
    
    val_data = None
    if args.val_data:
        print(f"Loading validation data: {args.val_data}")
        val_data = load_data_memmap(args.val_data)
        print(f"Validation data size: {len(val_data):,} tokens")
    
    # Initialize model
    print("\nInitializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        eps=args.eps,
        device=device,
        dtype=dtype
    )
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint (if specified)
    start_iter = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resuming training from iteration {start_iter}")
    
    # Training loop
    print(f"\nStarting training... (from iteration {start_iter} to {args.max_iters})")
    print("=" * 80)
    
    model.train()
    train_losses = []
    start_time = time.time()
    
    for iter_num in range(start_iter, args.max_iters):
        # Get current learning rate
        lr = get_lr_cosine_schedule(
            iter_num,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters
        )
        
        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch data
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        
        # Forward pass
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Log loss
        train_losses.append(loss.item())
        
        # Periodic logging
        if (iter_num + 1) % args.log_interval == 0:
            avg_loss = np.mean(train_losses[-args.log_interval:])
            elapsed = time.time() - start_time
            tokens_per_sec = (iter_num + 1 - start_iter) * args.batch_size * args.context_length / elapsed
            
            print(f"Iter {iter_num + 1}/{args.max_iters} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens/s: {tokens_per_sec:.0f}")
            
            if args.use_wandb:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/learning_rate': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'iter': iter_num + 1
                })
        
        # Periodic evaluation
        if (iter_num + 1) % args.eval_interval == 0:
            print("\n" + "=" * 80)
            print("Evaluating...")
            losses = estimate_loss(
                model, train_data, val_data,
                args.eval_iters, args.batch_size,
                args.context_length, device
            )
            
            print(f"Iter {iter_num + 1} | Train Loss: {losses['train']:.4f}", end="")
            if 'val' in losses:
                print(f" | Val Loss: {losses['val']:.4f}")
            else:
                print()
            print("=" * 80 + "\n")
            
            if args.use_wandb:
                log_dict = {
                    'eval/train_loss': losses['train'],
                    'iter': iter_num + 1
                }
                if 'val' in losses:
                    log_dict['eval/val_loss'] = losses['val']
                wandb.log(log_dict)
            
            model.train()
        
        # Periodic checkpoint saving
        if (iter_num + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num + 1}.pt"
            print(f"\nSaving checkpoint: {checkpoint_path}")
            save_checkpoint(model, optimizer, iter_num + 1, checkpoint_path)
            print()
    
    # Training finished
    print("\n" + "=" * 80)
    print("Training completed!")
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    print(f"Saving final checkpoint: {final_checkpoint_path}")
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    
    # Final evaluation
    if val_data is not None:
        print("\nFinal evaluation...")
        losses = estimate_loss(
            model, train_data, val_data,
            args.eval_iters, args.batch_size,
            args.context_length, device
        )
        print(f"Final training loss: {losses['train']:.4f}")
        print(f"Final validation loss: {losses['val']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'final/train_loss': losses['train'],
                'final/val_loss': losses['val']
            })
    
    if args.use_wandb:
        wandb.finish()
    
    print("=" * 80)


if __name__ == '__main__':
    main() 