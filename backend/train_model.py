import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from tqdm import tqdm
import os
import numpy as np
import argparse

# Enable mixed precision for better performance
torch.backends.cudnn.benchmark = True

# Constants
SIZE = 512
COLOR_DEVIATION = 0.01
BATCH_SIZE = 16

class AddNoise(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level
    
    def __call__(self, img):
        img_tensor = transforms.functional.to_tensor(img)
        noise = torch.rand_like(img_tensor) * self.noise_level
        noisy_img = img_tensor + noise
        return transforms.functional.to_pil_image(noisy_img)

# Define image transformations with data augmentation
image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=SIZE),
        AddNoise(0.01),
        transforms.ColorJitter(
            brightness=(1.0-COLOR_DEVIATION, 1.0+COLOR_DEVIATION),
            contrast=(1.0-COLOR_DEVIATION, 1.0+COLOR_DEVIATION),
            saturation=(1.0-COLOR_DEVIATION, 1.0+COLOR_DEVIATION),
            hue=(-1*COLOR_DEVIATION, COLOR_DEVIATION)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'validation': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

def prepare_data(dataset_path, batch_size=BATCH_SIZE):
    """
    Prepare data loaders for training, validation, and testing
    
    Args:
        dataset_path: Path to the dataset directory
        batch_size: Batch size for DataLoader
        
    Returns:
        train_data: DataLoader for training
        validation_data: DataLoader for validation
        test_data: DataLoader for testing
        class_names: List of class names
    """
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'validation')
    test_dir = os.path.join(dataset_path, 'test')
    
    # Load data from directories
    data = {
        'train': datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
        'validation': datasets.ImageFolder(root=val_dir, transform=image_transforms['validation']),
        'test': datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])
    }
    
    # Create data loaders
    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_data = DataLoader(data['validation'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    class_names = data['train'].classes
    
    print(f"Training samples: {len(data['train'])}")
    print(f"Validation samples: {len(data['validation'])}")
    print(f"Test samples: {len(data['test'])}")
    print(f"Classes: {class_names}")
    
    return train_data, validation_data, test_data, class_names

def create_model(num_classes=2):
    """
    Create a MobileNetV3Large model for glaucoma detection
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        
    Returns:
        model: PyTorch model
    """
    model = models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(1280, num_classes)
    return model

def train_model(model, train_data, validation_data, criterion, optimizer, scheduler, device, num_epochs=10):
    """
    Train the model with mixed precision
    
    Args:
        model: PyTorch model
        train_data: Training data loader
        validation_data: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs
        
    Returns:
        model: Trained model with best weights
    """
    model = model.to(device)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        epoch_progress = tqdm(total=len(train_data), desc=f'Epoch {epoch}/{num_epochs - 1}', position=0, leave=True)
        
        for phase in ['train', 'validation']:
            running_loss = 0.0
            running_corrects = 0
            
            if phase == 'train':
                model.train()
                dataloader = train_data
            else:
                model.eval()
                dataloader = validation_data
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                if phase == 'train':
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                epoch_progress.set_postfix(phase=phase, 
                                          loss=running_loss / ((epoch_progress.n + 1) * inputs.size(0)), 
                                          acc=running_corrects.double() / ((epoch_progress.n + 1) * inputs.size(0)))
                epoch_progress.update()
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        epoch_progress.close()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, test_data, device):
    """
    Evaluate the model on test data
    
    Args:
        model: PyTorch model
        test_data: Test data loader
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        test_acc: Test accuracy
    """
    model.eval()
    test_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    
    test_acc = test_corrects.double() / len(test_data.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')
    return test_acc

def save_model(model, model_path):
    """
    Save the model weights
    
    Args:
        model: PyTorch model
        model_path: Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main(args):
    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_data, validation_data, test_data, class_names = prepare_data(args.dataset_path, args.batch_size)
    
    # Create model
    model = create_model(num_classes=len(class_names))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Define learning rate scheduler - reduce LR after 3 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Train model
    model = train_model(
        model, train_data, validation_data, criterion, optimizer, scheduler, 
        device, num_epochs=args.epochs
    )
    
    # Evaluate model
    test_acc = evaluate_model(model, test_data, device)
    
    # Save model
    save_model(model, args.model_path)
    
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train glaucoma detection model')
    parser.add_argument('--dataset_path', type=str, default='./eyepac-light-v2-512-jpg', 
                        help='Path to the dataset directory')
    parser.add_argument('--model_path', type=str, default='./models/glaucoma_model.pth', 
                        help='Path to save the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, 
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=6, 
                        help='Number of epochs')
    
    args = parser.parse_args()
    main(args)
