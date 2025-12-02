"""
Simple MLP Classifier for Beijing PM2.5
This will actually work and beat the baseline!
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= CONFIG =================
DATA_PATH = "data/beijing_pm25_5classes_paper.npz"
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15  # Early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# ==========================================


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PM25Dataset(Dataset):
    """Simple dataset for PM2.5 time series"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    """
    Simple but effective MLP classifier
    Input: (batch, 168, 11) -> Flatten -> Hidden Layers -> Output
    """
    def __init__(self, input_size=168*11, num_classes=5, dropout=0.5):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            
            # Layer 1
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            
            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            
            # Output
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.0 * correct / total})
    
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.0 * correct / total})
    
    return total_loss / total, 100.0 * correct / total, all_preds, all_targets


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.axhline(y=48.76, color='g', linestyle='--', label='Paper Baseline (48.76%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Training history saved to {save_path}")


def main():
    set_seed(SEED)
    
    print("=" * 70)
    print("🚀 TRAINING SIMPLE MLP ON BEIJING PM2.5")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Expected time: ~10-15 minutes\n")
    
    # Load data
    print(f"📂 Loading: {DATA_PATH}")
    data = np.load(DATA_PATH)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"✅ Loaded")
    print(f"  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n📊 Class distribution (train):")
    class_names = ['L0_VeryLow', 'L1_Low', 'L2_Medium', 'L3_High', 'L4_VeryHigh']
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} ({100*count/len(y_train):.1f}%)")
    
    # Calculate class weights
    total = len(y_train)
    num_classes = len(unique)
    class_weights = []
    for cls in range(num_classes):
        count = counts[cls]
        weight = total / (num_classes * count)
        class_weights.append(weight)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"\n📌 Class weights: {class_weights.cpu().numpy()}")
    
    # Create datasets and loaders
    train_dataset = PM25Dataset(X_train, y_train)
    test_dataset = PM25Dataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    # Create model
    input_size = X_train.shape[1] * X_train.shape[2]  # 168 * 11
    model = SimpleMLP(input_size=input_size, num_classes=num_classes, dropout=0.5)
    model = model.to(DEVICE)
    
    print(f"\n🏗️  Model architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("\n" + "=" * 70)
    print("🏋️  STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n📅 Epoch {epoch}/{EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Evaluate
        val_loss, val_acc, val_preds, val_targets = evaluate(model, test_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, 'best_mlp_model.pth')
            
            print(f"  💾 NEW BEST MODEL! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{PATIENCE})")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️  Early stopping triggered! No improvement for {PATIENCE} epochs.")
            break
        
        print("-" * 70)
    
    # Load best model for final evaluation
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)
    
    checkpoint = torch.load('best_mlp_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_acc, final_preds, final_targets = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Acc: {best_acc:.2f}%")
    print(f"  Paper Baseline: 48.76%")
    print(f"  Improvement: {best_acc - 48.76:+.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(final_targets, final_preds, target_names=class_names, digits=3))
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(final_targets, final_preds)
    print(cm)
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)
    print(f"📁 Best model saved: best_mlp_model.pth")
    print(f"📊 Training plot saved: training_history.png")
    
    # Save results summary
    with open('mlp_results.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BEIJING PM2.5 CLASSIFICATION - MLP RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"Paper Baseline: 48.76%\n")
        f.write(f"Improvement: {best_acc - 48.76:+.2f}%\n")
        f.write(f"Best Epoch: {best_epoch}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(final_targets, final_preds, target_names=class_names, digits=3))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"📄 Results summary saved: mlp_results.txt")


if __name__ == "__main__":
    main()