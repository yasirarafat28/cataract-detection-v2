import json
import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models


import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.metrics import confusion_matrix, classification_report

# Load VIA annotations
with open('dataset/image_annotations.json') as f:
    annotations = json.load(f)

# Define class names for cataract type
class_names = ['No Cataract', 'Immature Cataract', 'Mature Cataract']

# Define a custom dataset
class CataractDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None):
        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = list(self.annotations.keys())
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, self.annotations[img_name]['filename'])
        image = Image.open(img_path).convert('RGB')
        
        # Draw shapes from annotations (if needed)
        draw = ImageDraw.Draw(image)
        regions = self.annotations[img_name]['regions']
        for region in regions:
            shape_attrs = region['shape_attributes']
            if shape_attrs['name'] == 'circle':
                cx, cy, r = shape_attrs['cx'], shape_attrs['cy'], shape_attrs['r']
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline='blue')
            # Add more shapes if needed

        # # Get the label and additional attributes from annotations
        # region_attributes = regions[0]['region_attributes']
        # cataract_presence = region_attributes.get('cataract_presence', 'no')
        # color_of_lens = region_attributes.get('color_of_lens', 'transparent')
        # cataract_location = region_attributes.get('cataract_location', 'central')
        # pupil_visibility = region_attributes.get('pupil_visibility', 'fully')
        # label = int(region_attributes.get('label', '0'))
        
        
        # Initialize default values for attributes
        cataract_presence = 'no'
        color_of_lens = 'transparent'
        cataract_location = 'central'
        pupil_visibility = 'fully'
        label = 0
        
        # Check if regions list is not empty
        if regions:
            region_attributes = regions[0]['region_attributes']
            cataract_presence = region_attributes.get('cataract_presence', 'no')
            color_of_lens = region_attributes.get('color_of_lens', 'transparent')
            cataract_location = region_attributes.get('cataract_location', 'central')
            pupil_visibility = region_attributes.get('pupil_visibility', 'fully')
            label = int(region_attributes.get('label', '0'))
        
        # Convert labels and attributes to indices
        cataract_presence_idx = {'no': 0, 'yes': 1}[cataract_presence]
        color_of_lens_idx = {'transparent': 0,'yellowish': 1, 'brownish': 2}[color_of_lens]
        cataract_location_idx = {'central': 0, 'peripheral': 1, 'diffuse': 2,'none':4}[cataract_location]
        pupil_visibility_idx = {'fully': 0, 'partially': 1, 'not_visible': 2}[pupil_visibility]

        if self.transform:
            image = self.transform(image)
        
        return image, label, cataract_presence_idx, color_of_lens_idx, cataract_location_idx, pupil_visibility_idx

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 

# Split data into training and validation sets (e.g., 80% train, 20% val)
train_data = {k: v for i, (k, v) in enumerate(annotations.items()) if i % 5 != 0}
val_data = {k: v for i, (k, v) in enumerate(annotations.items()) if i % 5 == 0}


# Define the dataset and dataloaders
root_dir = 'images'
# dataset = CataractDataset(annotations, root_dir, transform=data_transforms['train'])
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


train_dataset = CataractDataset(train_data, root_dir, transform=data_transforms['train'])
val_dataset = CataractDataset(val_data, root_dir, transform=data_transforms['val'])


train_size = len(train_dataset)
val_size = len(val_dataset)
print("train_size",train_size)
print("val_size",val_size)
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=True)
}
dataset_sizes = {'train': train_size, 'val': val_size}


# Section 2


# Load pretrained model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Adjust the final layer to output the main classification and auxiliary outputs
model.fc = nn.Linear(num_ftrs, len(class_names))

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 25
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

        for inputs, labels, cataract_presence, color_of_lens, cataract_position, pupil_visibility in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # If you want to use additional attributes, you can concatenate them with inputs or use as auxiliary targets
            # For example, you can create an auxiliary loss if needed.

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



# Section 3


# Function to evaluate model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for  inputs, labels, cataract_presence, color_of_lens, cataract_position, pupil_visibility in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return confusion_matrix(all_labels, all_preds), classification_report(all_labels, all_preds, target_names=class_names)

# Evaluate on validation set
conf_matrix, class_report = evaluate_model(model, dataloaders['val'])
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)


# Save the model
torch.save(model.state_dict(), 'cataract_detection_model.pth')

