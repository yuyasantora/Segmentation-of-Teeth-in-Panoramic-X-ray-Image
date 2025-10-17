"""
PyTorch training script for teeth segmentation
GPU-accelerated version
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNET


class TeethDataset(Dataset):
    """Dataset class for teeth X-ray images and masks"""

    def __init__(self, images, masks, transform=None):
        """
        Args:
            images: numpy array of shape (N, H, W, C)
            masks: numpy array of shape (N, H, W, C)
            transform: optional transforms
        """
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert from (H, W, C) to (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        acc = (predicted == masks).float().mean()
        running_acc += acc.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Statistics
            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            acc = (predicted == masks).float().mean()
            running_acc += acc.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def train_model(x_train, y_train, x_test, y_test,
                batch_size=8, epochs=150, learning_rate=0.001,
                device='cuda', save_path='unet_teeth_model.pth'):
    """
    Main training function

    Args:
        x_train: Training images (N, H, W, C)
        y_train: Training masks (N, H, W, C)
        x_test: Test images
        y_test: Test masks
        batch_size: Batch size for training
        epochs: Number of epochs
        learning_rate: Learning rate for optimizer
        device: 'cuda' or 'cpu'
        save_path: Path to save the model

    Returns:
        model: Trained model
        history: Training history
    """

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_dataset = TeethDataset(x_train, y_train)
    test_dataset = TeethDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # Create model
    model = UNET(input_shape=(512, 512, 1), last_activation='sigmoid')
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"Model saved! (Best val loss: {best_val_loss:.4f})")

    print("\nTraining complete!")

    # Plot training history
    plot_history(history)

    return model, history


def plot_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training history plot saved as 'training_history.png'")


def load_model(model_path, device='cuda'):
    """
    Load a saved model

    Args:
        model_path: Path to the saved model
        device: Device to load the model on

    Returns:
        model: Loaded model
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = UNET(input_shape=(512, 512, 1), last_activation='sigmoid')

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Best val loss: {checkpoint['val_loss']:.4f}")

    return model


def predict(model, image, device='cuda'):
    """
    Make prediction on a single image

    Args:
        model: Trained model
        image: Input image (H, W, C) numpy array
        device: Device to run on

    Returns:
        prediction: Predicted mask (H, W) numpy array
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    # Convert back to numpy
    prediction = output.cpu().squeeze().numpy()

    return prediction


if __name__ == "__main__":
    # Example usage
    print("This is a training module. Import it in your main script.")
    print("\nExample usage:")
    print("""
    from train_pytorch import train_model, load_model, predict

    # Train
    model, history = train_model(
        x_train, y_train, x_test, y_test,
        batch_size=8,
        epochs=150,
        device='cuda'
    )

    # Load
    model = load_model('unet_teeth_model.pth', device='cuda')

    # Predict
    prediction = predict(model, test_image, device='cuda')
    """)
