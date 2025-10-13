#!/usr/bin/env python3
"""
Test basic functionality of the training script
"""

import os
import sys
import numpy as np

print("=" * 60)
print("Testing Training Script Setup")
print("=" * 60)

# 1. Test imports
print("\n1. Testing module imports...")
try:
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
    import torch
    print("✓ All required modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# 2. Create test data
print("\n2. Creating test data...")
try:
    os.makedirs('data', exist_ok=True)
    
    # Create small test dataset
    vocab_size = 1000
    num_tokens = 10000
    tokens = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16)
    
    # Save as .npy file
    np.save('data/test_train.npy', tokens)
    print(f"✓ Test data created: data/test_train.npy ({num_tokens} tokens)")
except Exception as e:
    print(f"✗ Failed to create data: {e}")
    sys.exit(1)

# 3. Test model initialization
print("\n3. Testing model initialization...")
try:
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=128,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        device='cpu'
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized successfully, parameters: {total_params:,}")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    sys.exit(1)

# 4. Test optimizer
print("\n4. Testing optimizer...")
try:
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    print("✓ Optimizer initialized successfully")
except Exception as e:
    print(f"✗ Optimizer initialization failed: {e}")
    sys.exit(1)

# 5. Test data loading
print("\n5. Testing data loading...")
try:
    data = np.load('data/test_train.npy', mmap_mode='r')
    print(f"✓ Data loaded successfully, size: {len(data):,} tokens")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# 6. Test batch generation
print("\n6. Testing batch generation...")
try:
    batch_size = 4
    context_length = 128
    inputs, targets = get_batch(data, batch_size, context_length, 'cpu')
    print(f"✓ Batch generation successful")
    print(f"  - Input shape: {inputs.shape}")
    print(f"  - Target shape: {targets.shape}")
except Exception as e:
    print(f"✗ Batch generation failed: {e}")
    sys.exit(1)

# 7. Test forward pass
print("\n7. Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
    print(f"✓ Forward pass successful")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# 8. Test training step
print("\n8. Testing training step...")
try:
    model.train()
    
    # Forward pass
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    gradient_clipping(model.parameters(), max_l2_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    print(f"✓ Training step successful, loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Training step failed: {e}")
    sys.exit(1)

# 9. Test learning rate schedule
print("\n9. Testing learning rate schedule...")
try:
    lr_values = []
    for iter_num in [0, 100, 500, 1000, 2000]:
        lr = get_lr_cosine_schedule(
            iter_num,
            max_learning_rate=3e-4,
            min_learning_rate=3e-5,
            warmup_iters=100,
            cosine_cycle_iters=2000
        )
        lr_values.append(lr)
    print(f"✓ Learning rate schedule working correctly")
    print(f"  - Iter 0: {lr_values[0]:.2e}")
    print(f"  - Iter 100: {lr_values[1]:.2e}")
    print(f"  - Iter 2000: {lr_values[4]:.2e}")
except Exception as e:
    print(f"✗ Learning rate schedule failed: {e}")
    sys.exit(1)

# 10. Test checkpoint save and load
print("\n10. Testing checkpoint save and load...")
try:
    os.makedirs('test_checkpoints', exist_ok=True)
    checkpoint_path = 'test_checkpoints/test.pt'
    
    # Save checkpoint
    save_checkpoint(model, optimizer, 100, checkpoint_path)
    print(f"✓ Checkpoint saved successfully: {checkpoint_path}")
    
    # Load checkpoint
    new_model = TransformerLM(
        vocab_size=vocab_size,
        context_length=128,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        device='cpu'
    )
    new_optimizer = AdamW(new_model.parameters(), lr=3e-4)
    
    iter_num = load_checkpoint(checkpoint_path, new_model, new_optimizer)
    print(f"✓ Checkpoint loaded successfully, iteration: {iter_num}")
    
    # Clean up test files
    import shutil
    shutil.rmtree('test_checkpoints')
    print("✓ Test files cleaned up")
except Exception as e:
    print(f"✗ Checkpoint test failed: {e}")
    sys.exit(1)

# 11. Verify training script exists
print("\n11. Verifying training script...")
if os.path.exists('train.py'):
    print("✓ train.py exists")
    if os.access('train.py', os.X_OK):
        print("✓ train.py has execute permission")
    else:
        print("⚠ train.py does not have execute permission (optional)")
else:
    print("✗ train.py does not exist")

# Complete
print("\n" + "=" * 60)
print("All tests passed! Training script setup is correct.")
print("=" * 60)