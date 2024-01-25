import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

# Define a transform to resize the images to 20x20
transform = transforms.Compose([
    transforms.Resize((20, 20)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Load MNIST dataset
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)

# Create a directory to save images
os.makedirs('test', exist_ok=True)

# Function to save images
def save_image(tensor, label, index):
    image = transforms.ToPILImage()(tensor).convert("L")
    image.save(f'test/{label}_{index}.png')

# Counters for each digit
digit_counters = [0] * 10

# Process and save images
for images, labels in train_loader:
    label = labels.item()
    if digit_counters[label] < 10:
        save_image(images[0], label, digit_counters[label])
        digit_counters[label] += 1
    # Check if we have enough of each digit
    if all(x >= 5 for x in digit_counters):
        break

print("Images saved.")
