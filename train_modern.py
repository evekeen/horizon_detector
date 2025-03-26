import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from horizon_dataset import create_data_loaders
from horizon_model import HorizonNet, HorizonNetLight
from trainer import Trainer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train horizon detection model with modern training framework')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model', type=str, default='light', choices=['full', 'light'], help='model type: full (ResNet50) or light (MobileNetV3)')
    parser.add_argument('--scheduler', type=str, default='one_cycle', 
                        choices=['step', 'cosine', 'plateau', 'one_cycle', 'warmup_cosine'],
                        help='learning rate scheduler to use')
    parser.add_argument('--early-stopping', type=int, default=10, help='patience for early stopping')
    args = parser.parse_args()
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MPS not available, using {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create data loaders
    csv_file = 'horizon_data.csv'
    img_dir = 'images'
    batch_size = args.batch_size
    
    print("\n" + "=" * 50)
    print("HORIZON LINE DETECTION MODEL TRAINING")
    print("=" * 50)
    
    print("\n[1/5] Loading and preparing dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        csv_file, img_dir, batch_size=batch_size
    )
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Train samples: {len(train_loader.sampler)}")
    print(f"  - Validation samples: {len(val_loader.sampler)}")
    print(f"  - Test samples: {len(test_loader.sampler)}")
    
    print("\n[2/5] Creating model architecture...")
    # Create model
    if args.model == 'full':
        model = HorizonNet(pretrained=True)
        model_name = 'HorizonNet'
    else:
        model = HorizonNetLight(pretrained=True)
        model_name = 'HorizonNetLight'
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  - Architecture: {model_name}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    print("\n[3/5] Setting up training configuration...")
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir="checkpoints",
        tensorboard_dir="runs"
    )
    
    # Configure learning rate scheduler
    scheduler_config = {}
    
    if args.scheduler == 'step':
        scheduler_config = {
            'step_size': 5,
            'gamma': 0.5
        }
    elif args.scheduler == 'cosine':
        scheduler_config = {
            'T_max': args.epochs,
            'eta_min': 1e-6
        }
    elif args.scheduler == 'plateau':
        scheduler_config = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 3,
            'verbose': True
        }
    elif args.scheduler == 'one_cycle':
        scheduler_config = {
            'max_lr': args.lr * 10,
            'epochs': args.epochs,
            'steps_per_epoch': len(train_loader),
            'pct_start': 0.3,
            'div_factor': 25.0,
            'final_div_factor': 1000.0
        }
    elif args.scheduler == 'warmup_cosine':
        scheduler_config = {
            'warmup_epochs': int(args.epochs * 0.1),  # 10% of total epochs
            'cosine_epochs': args.epochs - int(args.epochs * 0.1),
            'eta_min': 1e-6
        }
    
    trainer.set_scheduler(args.scheduler, **scheduler_config)
    
    print(f"✓ Training configuration set")
    print(f"  - Loss function: MSELoss")
    print(f"  - Optimizer: Adam (lr={args.lr})")
    print(f"  - Scheduler: {args.scheduler}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Device: {device}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n[4/5] Resuming from checkpoint: {args.resume}")
        trainer.resume_from_checkpoint(args.resume)
    
    print("\n[4/5] Training model...")
    start_time = time.time()
    
    # Train the model
    model, train_losses, val_losses = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
    print(f"  - Final training loss: {train_losses[-1]:.6f}")
    print(f"  - Final validation loss: {val_losses[-1]:.6f}")
    print(f"  - Best validation loss: {min(val_losses):.6f}")
    
    print("\n[5/5] Evaluating model on test set...")
    mean_avg_y_error, mean_roll_error = trainer.evaluate(test_loader)
    
    print(f"\n✓ Model evaluation complete")
    print(f"  - Mean Average Y Error: {mean_avg_y_error:.2f} pixels")
    print(f"  - Mean Roll Angle Error: {mean_roll_error:.2f} degrees")
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 50)

if __name__ == "__main__":
    main()
