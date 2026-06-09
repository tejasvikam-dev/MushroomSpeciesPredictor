import torch
from model import ConvNet # Imports the class from your other file!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(num_classes=78).to(device)
model.load_state_dict(torch.load("mushroom_model.pt", map_location=device, weights_only=True))
model.eval()
