import json
import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
        color_of_lens_idx = {'transparent': 0, 'yellowish': 1, 'brownish': 2}[color_of_lens]
        cataract_location_idx = {'central': 0, 'peripheral': 1, 'diffuse': 2, 'none': 4}[cataract_location]
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

# Define the dataset and dataloaders
root_dir = 'images'
dataset = CataractDataset(annotations, root_dir, transform=data_transforms['train'])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=True)
}
dataset_sizes = {'train': train_size, 'val': val_size}

# Custom model definition
class CataractModel(nn.Module):
    def __init__(self, num_classes):
        super(CataractModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer
        
        # Additional attributes processing
        self.fc1 = nn.Linear(num_ftrs + 4, 512)  # 4 additional attributes
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, images, attributes):
        x = self.resnet(images)
        x = torch.cat((x, attributes), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
model = CataractModel(num_classes=len(class_names))

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

        for inputs, labels, cataract_presence_idx, color_of_lens_idx, cataract_location_idx, pupil_visibility_idx in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            attributes = torch.stack((cataract_presence_idx, color_of_lens_idx, cataract_location_idx, pupil_visibility_idx), dim=1)
            attributes = attributes.type(torch.float32).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs, attributes)
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

# Function to evaluate model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, cataract_presence_idx, color_of_lens_idx, cataract_location_idx, pupil_visibility_idx in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            attributes = torch.stack((cataract_presence_idx, color_of_lens_idx, cataract_location_idx, pupil_visibility_idx), dim=1)
            attributes = attributes.type(torch.float32).to(device)
            outputs = model(inputs, attributes)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return confusion_matrix(all_labels, all_preds), classification_report(all_labels, all_preds, target_names=class_names)

# Evaluate on validation set
conf_matrix, class_report = evaluate_model(model, dataloaders['val'])
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)
