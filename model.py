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

# Download latest version of the mushroom dataset
path = kagglehub.dataset_download("dariobaumberger/combined-kaggle-mushrooms-dataset")
print("Path to dataset files:", path)

# Pointing to the images folder inside the downloaded dataset
data_directory = os.path.join(path, "images")

# Loading 100 images for visual inspection

images = []
label = []

# Stefan's part ---

# Only grabbing common image types so we dont accidentally load junk files
image_exts = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".svg", ".ico", ".raw", ".cr2",
    ".nef", ".orf", ".sr2", ".eps", ".jp2", ".j2k", ".jpf",
    ".jpx", ".jpm", ".mj2")

# Walking through every subfolder to find images and their class labels
for root, dirs, files in os.walk(data_directory):
    for file in files:
        # Making sure we are not loading the wrong file types if they exist
        if file.lower().endswith(image_exts):
            img_path = os.path.join(root, file)
            # Saving image and using parent folder name as the label for the class
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            # Using the subfolder (class folder) name as the label instead of the filename
            class_name = os.path.basename(root)
            label.append(class_name)
            
            if len(images) >= 100:
                break
    if len(images) >= 100:
        break


# Teja's part:

# Checking what sizes are in dataset before resizing
sizes = [img.size for img in images]
unique_sizes = set(sizes)
print(f"Unique image sizes found: {unique_sizes}")
print(f"Total unique sizes: {len(unique_sizes)}")

# Displaying all 100 images in a 10x10 grid for quick visual inspection
fig = plt.figure(figsize=(20, 20))
for idx, image in enumerate(images):
    plt1 = plt.subplot(10, 10, idx + 1)
    plt1.imshow(image)
    # Trimming long class names so they fit in the subplot title
    plt1.set_title(label[idx][:10], fontsize=6)
    plt1.axis('off')

plt.tight_layout()
plt.savefig("eda_grid.png", dpi=100)
plt.close()
print("Saved EDA grid to eda_grid.png")

# Splitting the data into TRAIN TEST VAL: 

# Using splitfolders to organize our raw images into train/val/test splits
# ratio is (train, val, test) - 70% train, 15% val, 15% test
output_split_dir = os.path.join(os.path.dirname(data_directory), "split_data")

if not os.path.exists(output_split_dir):
    print("Splitting data into train, test, and val folders...")
    splitfolders.ratio(
        data_directory,
        output=output_split_dir,
        seed=42,
        ratio=(0.7, 0.15, 0.15)
    )
    print(f"Data split saved to: {output_split_dir}")
else:
    print("Split folders already exist, skipping split step.")

# Setting up TRAIN/ TEST/ VAL: 

# using augmentation (Stefan Contributed)
train_transforms = v2.Compose([
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