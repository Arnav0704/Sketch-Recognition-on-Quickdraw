import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2

# Constants
CLASSES = [
    "apple", "banana", "book", "car", "cat", "chair", "cloud", "dog", "door", "eye",
    "face", "fish", "flower", "fork", "guitar", "hammer", "hat", "house", "key", "knife",
    "leaf", "lightning", "moon", "mountain", "mouse", "star", "sun", "table", "tree", "umbrella"
]

# Define the same model architecture used during training
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

# Prediction Pipeline
class SketchClassifier:
    def __init__(self, model_path, class_names, img_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.img_size = img_size
        
        # Load trained model
        self.model = SketchMLP_S2(
            img_size=img_size,
            patch_size=4,
            in_chans=1,
            num_classes=len(class_names))
        
        # Load weights AND move to device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)  # Add this line
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = self.transform(img)
        return img.unsqueeze(0).to(self.device)

    def predict(self, image_path, topk=3):
        # Preprocess image (now guaranteed 1 channel and 224x224)
        input_tensor = self.preprocess_image(image_path)
        
        # Verify input shape
        print(f"Input tensor shape: {input_tensor.shape}")  # Should be [1, 1, 224, 224]
        
        # Rest of prediction logic remains the same
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_probs, top_indices = torch.topk(probabilities, topk)
        return [{
            "class": self.class_names[i.item()],
            "confidence": p.item()
        } for p, i in zip(top_probs[0], top_indices[0])]

# Usage Example
if __name__ == "__main__":
    # Initialize classifier
    classifier = SketchClassifier(
        model_path="/mnt/d/model/quick_draw/SketchMLP-S2/best_sketchMLP_model.pth",
        class_names=CLASSES,  # Your CLASSES list
        img_size=224
    )

    # Make prediction
    test_images_path = "../../../Test/tf/test/src/test/test_images"
    for file_name in os.listdir(test_images_path):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(test_images_path, file_name)
            print(f"Predicting for {file_name}...")
            results = classifier.predict(image_path)
            cv2.imshow("Image", cv2.imread(image_path))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print("Top Predictions:")
            for idx, result in enumerate(results):
                print(f"{idx+1}. {result['class']}: {result['confidence']*100:.2f}%")
                
    # singl image prediction example
    # results = classifier.predict("../../../Test/tf/test/src/test/test_images/car.jpg")
    
    # print("Top Predictions:")
    # for idx, result in enumerate(results):
    #     print(f"{idx+1}. {result['class']}: {result['confidence']*100:.2f}%")