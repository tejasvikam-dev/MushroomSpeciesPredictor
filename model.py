import kagglehub
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from torchvision.transforms import v2 
import splitfolders
from torch.utils.data import dataloader

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


for idx, image in enumerate (images):
    plt1 = plt.subplot(10, 10, idx+1)
    plt1.imshow(image)
    plt1.set_title(label[idx])
    plt1.axis('off')
plt.tight_layout()
plt.show()

# extract images from subfolders

