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



def filter_eye(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 4)
    print('Number of detected eyes:', len(eyes))

    for idx, (x, y, w, h) in enumerate(eyes):
        eye_img = img[y:y+h, x:x+w]
        return eye_img
    
    return None

def crop_center(image, crop_width=80, crop_height=80):
    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Calculate center point
    center_x, center_y = image_width // 2, image_height // 2

    # Calculate the starting and ending coordinates for the crop
    start_x = max(center_x - crop_width // 2, 0)
    end_x = min(center_x + crop_width // 2, image_width)
    start_y = max(center_y - crop_height // 2, 0)
    end_y = min(center_y + crop_height // 2, image_height)

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return cropped_image

# Define image preprocessing function
def preprocess_image(image_file,model_cataract,filename):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(120),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Read and preprocess the image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    eye_img = filter_eye(img)
    # eye_img = img

    if eye_img is None:
        print(f"No eyes detected in {filename}.")
        return None, None

    eye_pil_img = Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
    eye_tensor = preprocess(eye_pil_img)
    # Save the preprocessed eye image
    save_image(eye_tensor, os.path.join("result", "processed_" + filename))
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
            result, confidence = preprocess_image(uploaded_file,model_cataract,filename)
            
            if result == "too blurry":
                st.write(f"Image {filename} is too blurry, please take another one.")
            elif result:
                st.write(f"Image {filename}, Predicted {result} class, confidence - {confidence}%")
            else:
                st.write(f"Image {filename}, could not be processed.")

# To run the streamlit app, save this code to a file (e.g., app.py) and run `streamlit run app.py` in your terminal.
