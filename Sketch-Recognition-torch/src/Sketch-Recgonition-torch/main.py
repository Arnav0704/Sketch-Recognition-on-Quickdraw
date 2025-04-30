import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Constants
CLASSES = [
    "apple", "banana", "book", "car", "cat", "chair", "cloud", "dog", "door", "eye",
    "face", "fish", "flower", "fork", "guitar", "hammer", "hat", "house", "key", "knife",
    "leaf", "lightning", "moon", "mountain", "mouse", "star", "sun", "table", "tree", "umbrella"
]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 128
IMAGE_SIZE = 224
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for storing datasets
def create_directories():
    os.makedirs("data/", exist_ok=True)
    os.makedirs("data/npy/", exist_ok=True)
    os.makedirs("data/processed/", exist_ok=True)
    for class_name in CLASSES:
        os.makedirs(f"data/processed/{class_name}", exist_ok=True)

# Download QuickDraw NPY dataset
def download_quickdraw_npy_dataset():
    print("Downloading QuickDraw NPY dataset for sketch recognition...")
    for class_name in tqdm(CLASSES):
        try:
            # QuickDraw dataset NPY URL format
            url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                # Save NPY file
                with open(f"data/npy/{class_name}.npy", 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded NPY file for class {class_name}")
            else:
                print(f"Failed to download {class_name} NPY. Status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error downloading {class_name}: {e}")

# Process NPY files to create normalized tensors or images if needed
def process_npy_files(max_samples_per_class=10000):
    print("Processing NPY files...")
    class_data = {}
    
    for class_name in tqdm(CLASSES):
        npy_path = f"data/npy/{class_name}.npy"
        if os.path.exists(npy_path):
            # Load NPY file - contains bitmap data in shape [N, 784]
            sketches = np.load(npy_path)
            
            # Limit samples per class
            if len(sketches) > max_samples_per_class:
                # Random sampling to get variety
                indices = np.random.choice(len(sketches), max_samples_per_class, replace=False)
                sketches = sketches[indices]
            
            # Reshape from 784 to 28x28
            sketches = sketches.reshape(-1, 28, 28)
            
            # Store processed data
            class_data[class_name] = sketches
            
            # Optionally save as images for visualization
            # for i, sketch in enumerate(sketches[:10]):  # Save first 10 for visualization
            #     img = Image.fromarray(sketch).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            #     img.save(f"data/processed/{class_name}/{i}.png")
                
    return class_data

# Custom Dataset class for NPY sketches
class QuickDrawNPYDataset(Dataset):
    def __init__(self, class_data, class_to_idx, transform=None):
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Build dataset from class_data dictionary
        for class_name, sketches in class_data.items():
            class_idx = class_to_idx[class_name]
            for sketch in sketches:
                self.samples.append(sketch)
                self.targets.append(class_idx)
        
        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sketch = self.samples[idx]
        label = self.targets[idx]
        
        # Resize from 28x28 to target size
        sketch_resized = np.array(Image.fromarray(sketch).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        
        # Convert to tensor and keep as single channel
        sketch_tensor = torch.from_numpy(sketch_resized).float() / 255.0
        sketch_tensor = sketch_tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
        
        if self.transform:
            sketch_tensor = self.transform(sketch_tensor)
            
        return sketch_tensor, label

# Data transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale
    ])
    
    return train_transform, test_transform

# Spatial Shift MLP Module
class SpatialShiftMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
        self.out_features = out_features  # Track output channels
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # [Rest of spatial shift operations remain the same]
        # Spatial-shift operation
        x_shifted = x.clone()
        
        # For grayscale, we can use different patterns for spatial shifting
        if C == 1:
            # Split the spatial dimensions into quadrants and shift differently
            h_mid, w_mid = H // 2, W // 2
            
            # Shift top-left quadrant right
            x_shifted[:, :, :h_mid, :-1] = x[:, :, :h_mid, 1:]
            
            # Shift top-right quadrant left
            x_shifted[:, :, :h_mid, 1:] = x[:, :, :h_mid, :-1]
            
            # Shift bottom-left quadrant down
            x_shifted[:, :, :-1, :w_mid] = x[:, :, 1:, :w_mid]
            
            # Shift bottom-right quadrant up
            x_shifted[:, :, 1:, w_mid:] = x[:, :, :-1, w_mid:]
        else:
            # Original shift pattern for multiple channels
            chunk_size = max(1, C // 4)
            
            # Split the channels and shift in different directions
            # For the cases where C < 4, we adjust accordingly
            if C >= 1:
                # Group 1: shift left
                end_idx = min(chunk_size, C)
                x_shifted[:, :end_idx, :, 1:] = x[:, :end_idx, :, :-1]
            
            if C >= 2:
                # Group 2: shift right
                end_idx = min(chunk_size*2, C)
                x_shifted[:, chunk_size:end_idx, :, :-1] = x[:, chunk_size:end_idx, :, 1:]
            
            if C >= 3:
                # Group 3: shift up
                end_idx = min(chunk_size*3, C)
                x_shifted[:, chunk_size*2:end_idx, 1:, :] = x[:, chunk_size*2:end_idx, :-1, :]
            
            if C >= 4:
                # Group 4: shift down
                x_shifted[:, chunk_size*3:, :-1, :] = x[:, chunk_size*3:, 1:, :]
        
        
        # Reshape for MLP: [B, C, H, W] -> [B, H*W, C]
        x_reshaped = x_shifted.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # MLP layers
        x_reshaped = self.fc1(x_reshaped)
        x_reshaped = self.act(x_reshaped)
        x_reshaped = self.drop1(x_reshaped)
        x_reshaped = self.fc2(x_reshaped)
        x_reshaped = self.drop2(x_reshaped)
        
        # Reshape back using updated channel count
        x_out = x_reshaped.reshape(B, H, W, self.out_features).permute(0, 3, 1, 2)
        
        return x_out

# Modified SketchMLP_S2 class
class SketchMLP_S2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=1, num_classes=30, 
                 embed_dim=96, depths=[2, 2, 6, 2], mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Build MLP stages with proper channel progression
        self.stages = nn.ModuleList()
        current_dim = embed_dim
        for i, depth in enumerate(depths):
            stage = nn.Sequential()
            for j in range(depth):
                # Calculate output dimension
                out_dim = current_dim * 2 if j == depth-1 and i < len(depths)-1 else current_dim
                
                stage.append(SpatialShiftMLP(
                    in_features=current_dim,
                    hidden_features=int(current_dim * mlp_ratio),
                    out_features=out_dim,
                    drop=drop_rate
                ))
                
                # Update current dimension
                if j == depth-1 and i < len(depths)-1:
                    current_dim = out_dim
                
                # Add normalization
                stage.append(nn.BatchNorm2d(out_dim))
                stage.append(nn.GELU())
            
            self.stages.append(stage)
            
            # Add downsampling
            if i < len(depths) - 1:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(current_dim, current_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(current_dim),
                    nn.GELU()
                ))
                
        # Final classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(current_dim, num_classes)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Process through MLP stages
        for stage in self.stages:
            x = stage(x)
        
        # Classification head
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        
        return x

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    model.to(DEVICE)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total
            })
        
        # Step scheduler
        scheduler.step()
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/total:.4f}, Train Acc: {100.*correct/total:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_sketchMLP_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
    
    return model

# Evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/total, 100.*correct/total

# Visualize some sample sketches from the dataset
def visualize_samples(class_data, num_samples=5, classes_to_show=5):
    plt.figure(figsize=(15, 10))
    
    for i, class_name in enumerate(list(class_data.keys())[:classes_to_show]):
        sketches = class_data[class_name][:num_samples]
        
        for j, sketch in enumerate(sketches):
            plt.subplot(classes_to_show, num_samples, i*num_samples + j + 1)
            plt.imshow(sketch, cmap='gray')
            plt.title(class_name if j == 0 else "")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_sketches.png')
    plt.close()

# Visualize predictions
def visualize_predictions(model, dataloader, num_images=10):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 8))
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(2, num_images//2, images_so_far)
                ax.axis('off')
                ax.set_title(f'Pred: {CLASSES[preds[j]]}\nTrue: {CLASSES[labels[j]]}')
                
                # Display the grayscale image
                img = inputs.cpu().data[j, 0].numpy()
                # Denormalize
                img = img * 0.5 + 0.5
                img = np.clip(img, 0, 1)
                
                ax.imshow(img, cmap='gray')
                
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig('prediction_results.png')
                    return

# Main execution
def main():
    print(f"Using device: {DEVICE}")
    
    # Create directories
    create_directories()
    
    # Download QuickDraw NPY dataset
    # download_quickdraw_npy_dataset()
    
    # Process NPY files
    class_data = process_npy_files(max_samples_per_class=10000)
    
    # Visualize some samples
    visualize_samples(class_data)
    
    # Create class to index mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
    
    # Data transforms
    train_transform, test_transform = get_transforms()
    
    # Create full dataset
    full_dataset = QuickDrawNPYDataset(class_data, class_to_idx, transform=None)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create custom datasets with proper transforms
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)
    
    train_dataset = TransformedSubset(train_dataset, train_transform)
    val_dataset = TransformedSubset(val_dataset, test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = SketchMLP_S2(
        img_size=IMAGE_SIZE,
        patch_size=4,
        in_chans=1,  # Single channel for grayscale
        num_classes=NUM_CLASSES,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        mlp_ratio=4.,
        drop_rate=0.1
    )
    
    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS)
    
    # Evaluate on validation set
    val_loss, val_acc = evaluate_model(trained_model, val_loader, criterion)
    print(f'Final Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    
    # Visualize some predictions
    visualize_predictions(trained_model, val_loader)
    
    print("Training complete!")

if __name__ == "__main__":
    main()