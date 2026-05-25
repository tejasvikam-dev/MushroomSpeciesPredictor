import kagglehub
import matplotlib.pyplot as plt
from PIL import Image
import os
import splitfolders
from torchvision.transforms import v2 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Download latest version
path = kagglehub.dataset_download("dariobaumberger/combined-kaggle-mushrooms-dataset")

print("Path to dataset files:", path)


data_directory = r'C:\Users\stefa\.cache\kagglehub\datasets\dariobaumberger\combined-kaggle-mushrooms-dataset\versions\2\images'
images = []
label = []

image_exts = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", 
    ".webp", ".heic", ".heif", ".svg", ".ico", ".raw", ".cr2", 
    ".nef", ".orf", ".sr2", ".eps", ".jp2", ".j2k", ".jpf", 
    ".jpx", ".jpm", ".mj2")


# extract images from subfolders

for root, dirs, files in os.walk(data_directory):
    for file in files:
#making sure that we are not displaying the wrong filetypes if they exist
     if file.lower().endswith(image_exts):
        img_path = os.path.join(root, file)
#saving image paths 
        img = Image.open(img_path)
        images.append(img)
        label.append(file)
        # Stop once we have 100 images
        if len(images) >= 100:
            break
    if len(images) >= 100:
        break


plt.figure(figsize=(20,20))
for idx, image in enumerate (images):
    plt1 = plt.subplot(10,10,idx+1)
    plt1.imshow(image)
    plt1.set_title(label[idx][:10]) # make sure label isn't too long
    plt1.axis('off')
plt.tight_layout()
plt.show()



# data augments 
IMG_SIZE = 224

train_tranforms = v2.Compose([
      v2.ToTensor(),
      v2.RandomHorizontalFlip(0.5),
      v2.RandomResizedCrop(IMG_SIZE),
      v2.RandomRotation(30),
      v2.Normalize(mean=[0.485,0.456, 0.406], std = [0.229,0.224,0.225])
])

test_transforms = v2.Compose([
   v2.ToTensor(),
   v2.Resize((IMG_SIZE, IMG_SIZE)),
   v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# split data
ouptut_dir = "data_split"

splitfolders.ratio(data_directory, output=output_dir, seed=42, ratio=(0.7, 0.15, 0.15))


train_data = ImageFolder(os.path.join(output_dir, 'train'), transform =  train_tranforms)
val_data = ImageFolder(os.path.join(output_dir, 'val'), transform =  test_transforms)
test_data = ImageFolder(os.path.join(output_dir, 'test'), transform =  test_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader =  DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


