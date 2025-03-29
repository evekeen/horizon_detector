import os
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from torch.utils.tensorboard import SummaryWriter
from checkpoint_manager import CheckpointManager

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir="checkpoints",
        tensorboard_dir="runs",
        max_checkpoints=5,
        use_wandb=True,
        wandb_project="horizon_detector",
        wandb_config=None,
        accelerator=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = None
        self.use_wandb = use_wandb
        self.accelerator = accelerator
        
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, max_checkpoints)
        self.writer = SummaryWriter(tensorboard_dir)
        
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # If not using accelerator, move model to device
        if self.accelerator is None:
            self.model.to(self.device)
        
        # Initialize wandb if requested and not using accelerator's wandb
        if self.use_wandb and self.accelerator is None:
            if not wandb.run:
                # Initialize wandb
                wandb.init(project=wandb_project, config=wandb_config)
                
                # Log model architecture details
                wandb.watch(self.model, log='all')
        
    def set_scheduler(self, scheduler_type, **kwargs):
        if scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=kwargs.get("step_size", 10),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get("T_max", 50),
                eta_min=kwargs.get("eta_min", 0)
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.1),
                patience=kwargs.get("patience", 5),
                verbose=kwargs.get("verbose", True)
            )
        elif scheduler_type == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=kwargs.get("max_lr", 0.01),
                total_steps=kwargs.get("total_steps", None),
                epochs=kwargs.get("epochs", None),
                steps_per_epoch=kwargs.get("steps_per_epoch", None),
                pct_start=kwargs.get("pct_start", 0.3),
                div_factor=kwargs.get("div_factor", 25.0),
                final_div_factor=kwargs.get("final_div_factor", 10000.0)
            )
        elif scheduler_type == "warmup_cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from torch.optim.lr_scheduler import LinearLR
            from torch.optim.lr_scheduler import SequentialLR
            
            warmup_epochs = kwargs.get("warmup_epochs", 5)
            cosine_epochs = kwargs.get("cosine_epochs", 45)
            
            linear_scheduler = LinearLR(
                self.optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=cosine_epochs, 
                eta_min=kwargs.get("eta_min", 0)
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[linear_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
        return self.scheduler
    
    def resume_from_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
            
        if checkpoint_path is None:
            print("No checkpoint found. Starting from scratch.")
            return False
            
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        
        if checkpoint:
            self.current_epoch = checkpoint.get('epoch', -1) + 1
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.lr_history = checkpoint.get('lr_history', [])
            return True
            
        return False
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch+1}", leave=False)
        
        for inputs, targets in train_pbar:
            # If not using accelerator, move data to device
            if self.accelerator is None:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            if self.accelerator:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            
            # Optimizer step
            self.optimizer.step()
            
            # Update OneCycleLR scheduler every batch if used
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Update statistics
            current_loss = loss.item()
            running_loss += current_loss * inputs.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({'batch_loss': f"{current_loss:.4f}"})
        
        epoch_train_loss = running_loss / len(self.train_loader.sampler)
        self.train_losses.append(epoch_train_loss)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/train', epoch_train_loss, self.current_epoch)
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        self.writer.add_scalar('Learning_rate', current_lr, self.current_epoch)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'train_loss': epoch_train_loss,
                'learning_rate': current_lr,
                'epoch': self.current_epoch
            })
        
        return epoch_train_loss
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        
        val_pbar = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch+1}", leave=False)
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                # If not using accelerator, move data to device
                if self.accelerator is None:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                current_loss = loss.item()
                running_loss += current_loss * inputs.size(0)
                
                # Update progress bar
                val_pbar.set_postfix({'batch_loss': f"{current_loss:.4f}"})
        
        epoch_val_loss = running_loss / len(self.val_loader.sampler)
        self.val_losses.append(epoch_val_loss)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/validation', epoch_val_loss, self.current_epoch)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val_loss': epoch_val_loss,
                'epoch': self.current_epoch
            })
        
        # Update scheduler if it's ReduceLROnPlateau
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(epoch_val_loss)
        
        return epoch_val_loss
    
    def train(self, num_epochs, early_stopping_patience=None, test_loader=None):
        os.makedirs('training_plots', exist_ok=True)
        
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
        
        # For the second plot - learning rate
        lr_line, = ax2.plot([], [], 'g-', label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        # Early stopping variables
        early_stopping_counter = 0
        best_val_loss = float('inf')
        
        # Lists to store test metrics
        test_avg_y_errors = []
        test_roll_errors = []
        
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{self.current_epoch + num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss = self.validate_epoch()
            
            # Test phase (if test_loader is provided)
            if test_loader is not None:
                print(f"Testing model after epoch {epoch+1}...")
                mean_avg_y_error, mean_roll_error = self.evaluate(test_loader)
                test_avg_y_errors.append(mean_avg_y_error)
                test_roll_errors.append(mean_roll_error)
                
                # Log test metrics to tensorboard
                self.writer.add_scalar('Test/mean_avg_y_error', mean_avg_y_error, epoch)
                self.writer.add_scalar('Test/mean_roll_error', mean_roll_error, epoch)
                
                # Log test metrics to wandb
                if self.use_wandb:
                    wandb.log({
                        'test_avg_y_error': mean_avg_y_error,
                        'test_roll_error': mean_roll_error,
                        'epoch': epoch
                    })
            
            # Update scheduler (except for OneCycleLR and ReduceLROnPlateau)
            if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR) and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{self.current_epoch + num_epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.checkpoint_manager.save_best_model(self.model, val_loss, epoch)
                print("✓ Saved best model!")
                # Reset early stopping counter
                early_stopping_counter = 0
            else:
                # Increment early stopping counter
                if early_stopping_patience is not None:
                    early_stopping_counter += 1
                    print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch,
                self.train_losses, self.val_losses, self.best_val_loss,
                metrics={'lr_history': self.lr_history}
            )
            
            # Update the plots
            epochs = list(range(1, epoch + 2))
            
            # Update line plots
            train_line.set_data(epochs, self.train_losses)
            val_line.set_data(epochs, self.val_losses)
            lr_line.set_data(epochs, self.lr_history)
            
            # Adjust axes limits
            ax1.set_xlim(1, max(self.current_epoch + num_epochs, epoch + 1))
            if self.train_losses and self.val_losses:
                y_min = min(min(self.train_losses), min(self.val_losses)) * 0.9
                y_max = max(max(self.train_losses), max(self.val_losses)) * 1.1
                ax1.set_ylim(y_min, y_max)
            
            if self.lr_history:
                ax2.set_xlim(1, max(self.current_epoch + num_epochs, epoch + 1))
                ax2.set_ylim(min(self.lr_history) * 0.9, max(self.lr_history) * 1.1)
            
            # Draw the updated plots
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Save the current plot
            plot_dir = 'training_plots'
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'epoch_{epoch+1}.png'))
            
            # Print progress bar
            progress = (epoch + 1 - self.current_epoch) / num_epochs * 100
            progress_bar = f"[{'=' * int(progress // 2)}{' ' * (50 - int(progress // 2))}] {progress:.1f}%"
            print(progress_bar)
            
            # Check for early stopping
            if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Update current epoch
            self.current_epoch = epoch + 1
        
        # Turn off interactive mode
        plt.ioff()
        
        # Create a final plot
        self._create_final_plot()
        
        # Load the best model
        best_model_path = os.path.join("checkpoints", "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint.get('val_loss', 'unknown')}")
        
        # Close tensorboard writer
        self.writer.close()
        
        # Finish wandb run
        if self.use_wandb and wandb.run:
            # Log final model as an artifact
            model_artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}", 
                type="model",
                description=f"Trained horizon detection model - final validation loss: {min(self.val_losses):.6f}"
            )
            # Add the best model file to the artifact
            model_artifact.add_file(os.path.join("checkpoints", "best_model.pth"))
            wandb.log_artifact(model_artifact)
            
            # Upload plots
            wandb.log({"final_training_curve": wandb.Image(os.path.join('training_plots', 'training_results.png'))})
            
            # Finish the run
            wandb.finish()
        
        # Return test metrics if collected
        if test_loader is not None and test_avg_y_errors and test_roll_errors:
            return self.model, self.train_losses, self.val_losses, test_avg_y_errors, test_roll_errors
        else:
            return self.model, self.train_losses, self.val_losses
    
    def _create_final_plot(self):
        plt.figure(figsize=(15, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 1, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-o', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-o', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.lr_history, 'g-o', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs('training_plots', exist_ok=True)
        plt.savefig(os.path.join('training_plots', 'training_results.png'), dpi=300)
        
    def evaluate(self, test_loader):
        self.model.eval()
        
        avg_y_errors = []
        roll_angle_errors = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                # If not using accelerator, move data to device
                if self.accelerator is None:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # Denormalize the predictions and targets
                # Use dataset statistics to denormalize the z-score normalized values
                avg_y_mean = test_loader.dataset.avg_y_mean
                avg_y_std = test_loader.dataset.avg_y_std
                roll_mean = test_loader.dataset.roll_mean
                roll_std = test_loader.dataset.roll_std
                
                # Denormalize predictions using z-score statistics
                pred_avg_y = outputs[:, 0] * avg_y_std + avg_y_mean
                pred_roll = outputs[:, 1] * roll_std + roll_mean
                
                # Denormalize targets using z-score statistics
                true_avg_y = targets[:, 0] * avg_y_std + avg_y_mean
                true_roll = targets[:, 1] * roll_std + roll_mean
                
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
