import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np

# Define class names for cataract type
class_names = ['No Cataract', 'Immature Cataract', 'Mature Cataract']

# Load the model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('cataract_detection_model.pth'))
model.eval()

# Define image preprocessing function
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def imshow(tensor, title=None):
    # Convert the tensor to a NumPy array and denormalize it
    image = tensor.numpy().squeeze()
    image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

# Predict function
def predict(image_path):
    image = preprocess_image(image_path)
    imshow(image.squeeze(), title='Preprocessed Image')
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print("ddd",predicted)
        return class_names[predicted[0]]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    prediction = predict(image_path)
    print(f'Predicted cataract level: {prediction}')
