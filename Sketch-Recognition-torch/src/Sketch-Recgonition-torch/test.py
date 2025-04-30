import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

# Load the saved model
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    image_size = checkpoint['image_size']
    
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, image_size

# Preprocess the custom image
def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open the image using PIL
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Predict class for custom images
def predict_image(model, image_tensor, class_names, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Visualize image with prediction
def visualize_prediction(image_path, predicted_class):
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis("off")
    plt.show()

# Main function for testing custom images
def test_custom_image(model_path, image_path, device):
    model, class_names, image_size = load_model(model_path, device)
    image_tensor = preprocess_image(image_path, image_size)
    predicted_class = predict_image(model, image_tensor, class_names, device)
    visualize_prediction(image_path, predicted_class)

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "efficientnet_quickdraw_model.pth"  # Path to your saved model
image_path = "path/to/your/image.jpg"  # Replace with the path to your custom image

test_custom_image(model_path, image_path, device)
