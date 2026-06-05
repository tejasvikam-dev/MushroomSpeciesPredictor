import kagglehub
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from torchvision.transforms import v2
import splitfolders
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torchvision.models as models
import wandb

# Download latest version of the mushroom dataset
output_split_dir = r"C:\Users\stefa\OneDrive\Documents\GitHub\CMPM\MushroomSpeciesPredictor\data_split"

# Loading 100 images for visual inspection

images = []
label = []
image_exts = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", 
    ".webp", ".heic", ".heif", ".svg", ".ico", ".raw", ".cr2", 
    ".nef", ".orf", ".sr2", ".eps", ".jp2", ".j2k", ".jpf", 
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".svg", ".ico", ".raw", ".cr2",
    ".nef", ".orf", ".sr2", ".eps", ".jp2", ".j2k", ".jpf",
    ".jpx", ".jpm", ".mj2")
# Stefan's part ---
# 's part ---
print("Loading 100 images from the train split")
train_dir = os.path.join(output_split_dir, "train")


# Using os.walk to go through all folders and subfolders
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.lower().endswith(image_exts):
            img_path = os.path.join(root, file)
            
            with Image.open(img_path) as img:
                images.append(img.copy().convert("RGB"))
            
            folder_name = os.path.basename(root)
            label.append(folder_name)
            
            if len(images) >= 100:
                break
    if len(images) >= 100:
        break

    
    # Loop exactly 100 times to fill the 10x10 grid
# Teja's part:

# Checking what sizes are in dataset before resizing
sizes = [img.size for img in images]
unique_sizes = set(sizes)
print(f"Unique image sizes found: {unique_sizes}")
print(f"Total unique sizes: {len(unique_sizes)}")

# Displaying all 100 images in a 10x10 grid for  visual inspection
fig = plt.figure(figsize=(20, 20))
for idx, image in enumerate(images):
    plt1 = plt.subplot(10, 10, idx + 1)
    plt1.imshow(image)
    # Trimming long class names so they fit in the subplot title
    plt1.set_title(label[idx][:10], fontsize=6)
    plt1.axis('off')

plt.tight_layout()
plt.savefig(r"C:\Users\stefa\OneDrive\Documents\GitHub\CMPM\MushroomSpeciesPredictor\display.png", dpi=100)
print("Saved EDA grid to display.png")

# 2. Pop it up on your screen so you can see it with your own eyes!
plt.show() 

# 3. Close it out of memory
plt.close()
# Splitting the data into TRAIN TEST VAL: 

# Using splitfolders to organize our raw images into train/val/test splits
# ratio is (train, val, test) - 70% train, 15% val, 15% test


# Setting up TRAIN/ TEST/ VAL: 

# using augmentation (Stefan Contributed)
train_transforms = v2.Compose([
    v2.ToTensor(),
    v2.ToImage(),                                        # Converting PIL image into somethign the tensor can understand
    v2.ToDtype(torch.float32, scale=True),               # Scaling pixel values to (0, 1)
    v2.Resize((224, 224)),                               # Resizing all images to the same sizes
    v2.RandomHorizontalFlip(p=0.5),                      # Randomly flipping horizontally - as mushrooms can be found from any angle
    v2.RandomVerticalFlip(p=0.2),                        
    v2.RandomRotation(degrees=30),                       # Rotating up to 30 degrees - mushrooms grow at various angles
    v2.ColorJitter(brightness=0.3, contrast=0.3,         # Randomly changing lighting since photos are taken in different conditions
                   saturation=0.3, hue=0.1),
    v2.RandomGrayscale(p=0.1),                           # removing color
    v2.RandomPerspective(distortion_scale=0.3, p=0.3),  
    v2.Normalize(mean=[0.485, 0.456, 0.406],             # Normalizing using ImageNet stats since mushroom images are natural photos
                 std=[0.229, 0.224, 0.225]),
])

# Val/test transforms - no augmentation, just resize, tensor, and normalize
val_test_transforms = v2.Compose([
    v2.ToTensor(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),                               # Resizing to match train size
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])


# Using ImageFolder to create train, val, and test dataset objects

# allowing ImageFolder to reads the subfodders as labels for the classes
train_dataset = ImageFolder(
    root=os.path.join(output_split_dir, "train"),
    transform=train_transforms
)

val_dataset = ImageFolder(
    root=os.path.join(output_split_dir, "val"),
    transform=val_test_transforms
)

test_dataset = ImageFolder(
    root=os.path.join(output_split_dir, "test"),
    transform=val_test_transforms
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")
print(f"Test samples:  {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")


# Using DataLoaders for loading batches of data during training (Teja and Stefan worked together)

# shuffling training data each epoch
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,       # Shuffling train data so the model doesnt memorize order
    num_workers=0,
    pin_memory=True
)

# Using data loaders to feed validation data in order (no shuffle needed)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Using data loaders to load test data for final evaluation
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Looping through data

# Looping through train loader to check that inputs and outputs match up
print("\nChecking Train Loader... ")
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1} | Input shape: {inputs.shape} | Output (labels): {targets}")
    if batch_idx >= 2:   # Only checking the first 3 batches so we dont wait forever
        break

# Looping through val loader to make sure val batches are loading properly
print("\nChecking Val Loader...")
for batch_idx, (inputs, targets) in enumerate(val_loader):
    print(f"Batch {batch_idx + 1} | Input shape: {inputs.shape} | Output (labels): {targets}")
    if batch_idx >= 2:
        break

# Looping through test loader to make sure test data is structured same
print("\n Checking Test Loader...")
for batch_idx, (inputs, targets) in enumerate(test_loader):
    print(f"Batch {batch_idx + 1} | Input shape: {inputs.shape} | Output (labels): {targets}")
    if batch_idx >= 2:
        break

print("\nAll data loaders are verified.")




# establishing where processes will happen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







# Creating Convolution Layer


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 3 input channels 32 output
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        # 32 input 64 output
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)

        self.pool = nn.MaxPool2d(2,2)

        # channels * height * width
        self.fc1 = nn.Linear(64*56*56,512)
        self.fc2 = nn.Linear(512, num_classes)

        # 100 classes
        self.relu = nn.ReLU()

# forward function
    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        X = X.flatten(start_dim=1)
        X = self.relu(self.fc1(X))
        output = self.fc2(X)
        return output


num_classes = len(train_dataset.classes)
model = ConvNet(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
# Add this before setting num_epochs
wandb.init(project="MushroomSpeciesPredictor", name="CNN_run_1")
num_epochs = 5





print("\nStarting Training Loop")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = 100.0 * correct_train / total_train
    
    # validation check
    model.eval()
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_val += targets.size(0)
            correct_val += predicted.eq(targets).sum().item()
            
    epoch_val_acc = 100.0 * correct_val / total_val
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}%")
    
    # log metrics to WandB
    wandb.log({
        "train_loss": epoch_loss, 
        "train_accuracy": epoch_train_acc,
        "val_accuracy": epoch_val_acc
    })

print("Training complete.")

# final test evaluation
print("\nRunning Final Test Evaluation...")
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total_test += targets.size(0)
        correct_test += predicted.eq(targets).sum().item()

final_test_acc = 100.0 * correct_test / total_test
print(f"=== Final Test Accuracy: {final_test_acc:.2f}% ===")
wandb.log({"final_test_accuracy": final_test_acc})

# save the model state dictionary for deployment
torch.save(model.state_dict(), "mushroom_model.pt")
print("Model state saved to mushroom_model.pt")

wandb.finish()
