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
import cv2
import numpy as np
from utils import change_color_outside_circle
from utils import filter_eye,crop_center
# Define class labels
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



pp = transforms.Compose([
        
        transforms.Resize(256),
        transforms.CenterCrop(256),
    ])

# Define image preprocessing function
def preprocess_image(image_file,model_cataract,filename):
    preprocess = transforms.Compose([
        
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_file).convert('RGB')
    
    need_filter = True
    if need_filter:
        img = np.array(img)
        eye_img = filter_eye(img)
        if eye_img is None:
            print(f"No eyes detected in {filename}.")
            return None, None
        eye_img = crop_center(eye_img,128,128)
        # Resize the image to 256x256 after cropping
        ss= np.array(eye_img)
        eye_img=pp(Image.fromarray(eye_img))
        # eye_img = Image.fromarray(eye_img)

        
        if eye_img is None:
            print(f"No eyes detected in {filename}.")
            return None, None
    else:
        eye_img = img

    if eye_img is None:
        print(f"No eyes detected in {filename}.")
        return None, None
    
    
    # Convert the PIL Image to a NumPy array
    image_np = np.array(eye_img)

    # Apply the change_color_outside_circle function to the NumPy array
    image_np = change_color_outside_circle(image_np)

    # Convert the processed NumPy array back to a PIL Image for further processing
    eye_img = Image.fromarray(image_np)
    
    
    eye_tensor = preprocess(eye_img)
    
    
    # Save the preprocessed eye image
    # save_image(image_np, os.path.join("result", "processed_" + filename))
    save_path = os.path.join("result", f"processed_{filename}")
    transformed_image = transforms.ToPILImage()(image_np).convert("RGB")
    transformed_image.save(save_path)
    
    
    
    eye_tensor = eye_tensor.unsqueeze(0)  # Add batch dimension


    
    # Move the image tensor to the same device as the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eye_tensor = eye_tensor.to(device)
    model_cataract = model_cataract.to(device)
    
    outputs = model_cataract(eye_tensor)
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

uploaded_files = st.file_uploader("Upload eye images", accept_multiple_files=True, type=["jpg", "jpeg", "png"],help="Please put the iris or lens center of the image")

if st.button("Classify Images"):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        model_cataract = load_models(cataract_model)
        
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            result, confidence = preprocess_image(uploaded_file,model_cataract,filename)
            
            if result == "too blurry":
                st.write(f"Image {filename} is too blurry, please take another one.")
            # elif confidence <75:
            #     st.write(f"Image {filename}, could not be processed. Please try another image")
            elif result:
                st.write(f"Image {filename}, Predicted {result} class, confidence - {confidence}%")
            else:
                st.write(f"Image {filename}, could not be processed.")

# To run the streamlit app, save this code to a file (e.g., app.py) and run `streamlit run app.py` in your terminal.
