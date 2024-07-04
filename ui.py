import streamlit as st
from glob import glob
import os
import torch
from PIL import Image
from torchvision import transforms,models
from torchvision.utils import save_image

import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

# Define class labels
class_labels = {0: "healthy", 1: 'cataract-mild', 2: 'cataract-heavy'}
class_names = ['No Cataract', 'Immature Cataract', 'Mature Cataract']

# Load models
@st.cache_resource
def load_models(cataract_path):
    
    # Load the model
    model_cataract = models.resnet18(pretrained=False)
    num_ftrs = model_cataract.fc.in_features
    model_cataract.fc = nn.Linear(num_ftrs, len(class_names))
    model_cataract.load_state_dict(torch.load(cataract_path))
    model_cataract.eval()
    return model_cataract

# Define image preprocessing function
def preprocess_image(image_path,model_cataract):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    # save_image(image, os.path.join(output_path, "processed_" + filename))
    outputs = model_cataract(image)
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    label = class_names[predicted[0]]
    confidence = int(confidence.item() * 100)  # Convert to percentage
    return label,confidence


# Streamlit UI
st.title("Cataract Classification Model")

# data_path = st.text_input("Path to folder with eye images", "./data")
data_path="./data"
cataract_model = "cataract_detection_model.pth"
output_path="./result"
threshold = 0.01

uploaded_files = st.file_uploader("Upload eye images", accept_multiple_files=True, type=["jpg", "jpeg", "png"],help="Please put the irish or lens center of the image")

if st.button("Classify Images"):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        model_cataract = load_models(cataract_model)
        
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            result, confidence = preprocess_image(uploaded_file,model_cataract)
            
            if result == "too blurry":
                st.write(f"Image {filename} is too blurry, please take another one.")
            elif result:
                st.write(f"Image {filename}, Predicted {result} class, confidence - {confidence}%")
            else:
                st.write(f"Image {filename}, could not be processed.")

# To run the streamlit app, save this code to a file (e.g., app.py) and run `streamlit run app.py` in your terminal.
