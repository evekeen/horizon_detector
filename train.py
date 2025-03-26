import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import matplotlib.ticker as ticker
from horizon_dataset import HorizonDataset, create_data_loaders
from horizon_model import HorizonNet, HorizonNetLight

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='mps'):
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    # Set up the figure for real-time plotting
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Initialize plots
    train_line, = ax1.plot([], [], 'b-', label='Training Loss')
    val_line, = ax1.plot([], [], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # For the second plot - loss trend
    bar_width = 0.35
    bar_positions = np.arange(2)
    bars = ax2.bar(bar_positions, [0, 0], bar_width, label=['Train', 'Val'])
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(['Train', 'Val'])
    ax2.set_ylabel('Current Loss')
    ax2.set_title('Current Epoch Loss')
    ax2.grid(True, axis='y')
    
    # For saving the figure
    os.makedirs('training_plots', exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, targets in train_pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            # Update statistics
            current_loss = loss.item()
            running_loss += current_loss * inputs.size(0)
            batch_count += 1
            
            # Update progress bar
            train_pbar.set_postfix({'batch_loss': f"{current_loss:.4f}"})
        
        epoch_train_loss = running_loss / len(train_loader.sampler)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                current_loss = loss.item()
                running_loss += current_loss * inputs.size(0)
                
                # Update progress bar
                val_pbar.set_postfix({'batch_loss': f"{current_loss:.4f}"})
        
        epoch_val_loss = running_loss / len(val_loader.sampler)
        val_losses.append(epoch_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_horizon_model.pth')
            print("✓ Saved best model!")
        
        # Update the plots
        epochs = list(range(1, epoch + 2))
        
        # Update line plots
        train_line.set_data(epochs, train_losses)
        val_line.set_data(epochs, val_losses)
        
        # Update bar plots
        bars[0].set_height(epoch_train_loss)
        bars[1].set_height(epoch_val_loss)
        
        # Adjust axes limits
        ax1.set_xlim(1, max(num_epochs, epoch + 1))
        y_min = min(min(train_losses), min(val_losses)) * 0.9
        y_max = max(max(train_losses), max(val_losses)) * 1.1
        ax1.set_ylim(y_min, y_max)
        
        ax2.set_ylim(0, max(epoch_train_loss, epoch_val_loss) * 1.2)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        # Draw the updated plots
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Save the current plot
        plt.savefig(f'training_plots/epoch_{epoch+1}.png')
        
        # Print progress bar
        progress = (epoch + 1) / num_epochs * 100
        progress_bar = f"[{'=' * int(progress // 2)}{' ' * (50 - int(progress // 2))}] {progress:.1f}%"
        print(progress_bar)
    
    # Load the best model
    model.load_state_dict(torch.load('best_horizon_model.pth'))
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device='mps'):
    model.eval()
    
    avg_y_errors = []
    roll_angle_errors = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Denormalize the predictions and targets
            pred_avg_y = outputs[:, 0] * 5000.0
            pred_roll = outputs[:, 1] * 90.0
            
            true_avg_y = targets[:, 0] * 5000.0
            true_roll = targets[:, 1] * 90.0
            
            # Calculate errors
            avg_y_error = torch.abs(pred_avg_y - true_avg_y)
            roll_error = torch.abs(pred_roll - true_roll)
            
            avg_y_errors.extend(avg_y_error.cpu().numpy())
            roll_angle_errors.extend(roll_error.cpu().numpy())
    
    mean_avg_y_error = np.mean(avg_y_errors)
    mean_roll_error = np.mean(roll_angle_errors)
    
    print(f"Mean Average Y Error: {mean_avg_y_error:.2f} pixels")
    print(f"Mean Roll Angle Error: {mean_roll_error:.2f} degrees")
    
    return mean_avg_y_error, mean_roll_error

def plot_results(train_losses, val_losses):
    plt.ioff()  # Turn off interactive mode for final plot
    
    # Create a new figure for the final results
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Plot loss difference (train - val)
    plt.subplot(2, 1, 2)
    loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
    plt.plot(epochs, loss_diff, 'g-o', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Loss Difference (Train - Val)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Difference', fontsize=12)
    plt.grid(True)
    
    # Add annotations for min and max values
    min_idx = loss_diff.index(min(loss_diff))
    max_idx = loss_diff.index(max(loss_diff))
    plt.annotate(f'Min: {loss_diff[min_idx]:.4f}', 
                xy=(epochs[min_idx], loss_diff[min_idx]),
                xytext=(epochs[min_idx], loss_diff[min_idx] - 0.02),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)
    plt.annotate(f'Max: {loss_diff[max_idx]:.4f}', 
                xy=(epochs[max_idx], loss_diff[max_idx]),
                xytext=(epochs[max_idx], loss_diff[max_idx] + 0.02),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    plt.show()

def main():
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
    batch_size = 32
    
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
    model = HorizonNetLight(pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  - Architecture: HorizonNetLight")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    print("\n[3/5] Setting up training configuration...")
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    
    print(f"✓ Training configuration set")
    print(f"  - Loss function: MSELoss")
    print(f"  - Optimizer: Adam (lr=0.001)")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Device: {device}")
    
    print("\n[4/5] Training model...")
    start_time = time.time()
    
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=num_epochs, device=device
    )
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
    print(f"  - Final training loss: {train_losses[-1]:.6f}")
    print(f"  - Final validation loss: {val_losses[-1]:.6f}")
    print(f"  - Best validation loss: {min(val_losses):.6f}")
    
    print("\n[5/5] Evaluating model on test set...")
    mean_avg_y_error, mean_roll_error = evaluate_model(model, test_loader, device=device)
    
    print(f"\n✓ Model evaluation complete")
    print(f"  - Mean Average Y Error: {mean_avg_y_error:.2f} pixels")
    print(f"  - Mean Roll Angle Error: {mean_roll_error:.2f} degrees")
    
    print("\nGenerating final training plots...")
    plot_results(train_losses, val_losses)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 50)

if __name__ == "__main__":
    main()
