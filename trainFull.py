import os
import random
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from utils import change_color_outside_circle
import numpy as np

# Define class names for cataract type
class_names = ['No_Cataract', 'Immature_Cataract', 'Mature_Cataract']

# Define a custom dataset
class CataractDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', val_split=0.3, save_transformed=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = class_names
        self.images = []
        self.labels = []
        self.save_transformed = save_transformed
        self.save_dir = os.path.join(root_dir, 'preprocessed_images')
        
        if save_transformed:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Gather all images and labels
        image_paths = []
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_file in os.listdir(class_dir):
                image_paths.append((os.path.join(class_dir, img_file), idx))
        
        # Randomly shuffle the data
        random.shuffle(image_paths)
        
        # Split data into training and validation sets
        num_val = int(len(image_paths) * val_split)
        if split == 'train':
            image_paths = image_paths[num_val:]
        elif split == 'val':
            image_paths = image_paths[:num_val]

        # Unpack the filtered image paths and labels
        for path, label in image_paths:
            self.images.append(path)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Convert the PIL Image to a NumPy array
        image_np = np.array(image)

        # Apply the change_color_outside_circle function to the NumPy array
        image_np = change_color_outside_circle(image_np)

        # Convert the processed NumPy array back to a PIL Image for further processing
        image = Image.fromarray(image_np)
        original_image = image.copy()  # Copy original image for saving
        
        if self.transform:
            image = self.transform(image)
            

        if self.save_transformed:
            save_path = os.path.join(self.save_dir, f"processed_img_{idx}.png")
            transformed_image = transforms.ToPILImage()(image).convert("RGB")
            transformed_image.save(save_path)
        
        return image, label

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Assuming data directory structure
root_dir = 'data'

train_dataset = CataractDataset(root_dir, transform=data_transforms['train'], split='train', save_transformed=True)
val_dataset = CataractDataset(root_dir, transform=data_transforms['val'], split='val', save_transformed=True)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Load pretrained model and adjust final layer
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Setup loss function, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print('Training complete')
torch.save(model.state_dict(), 'cataract_detection_model.pth')
