import os
import torch
import glob
from datetime import datetime

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints", max_to_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, train_losses, val_losses, best_val_loss, metrics=None, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch:03d}_{timestamp}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth"))
        
        if len(checkpoints) <= self.max_to_keep:
            return
        
        checkpoints.sort(key=os.path.getmtime)
        for checkpoint in checkpoints[:-self.max_to_keep]:
            os.remove(checkpoint)
            print(f"Removed old checkpoint: {checkpoint}")
    
    def load_checkpoint(self, path, model, optimizer=None, scheduler=None):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return None
        
        checkpoint = torch.load(path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'epoch' in checkpoint:
            print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
            return checkpoint
        else:
            print(f"Loaded checkpoint from {path}")
            return None
    
    def get_latest_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth"))
        
        if not checkpoints:
            return None
        
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        return latest_checkpoint
    
    def save_best_model(self, model, val_loss, epoch, filename="best_model.pth"):
        best_model_path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }, best_model_path)
        
        print(f"Best model saved: {best_model_path}")
        return best_model_path
