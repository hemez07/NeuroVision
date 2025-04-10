import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess an image
image_path = "sample.jpg"  # <- Replace with your image path
image = Image.open(image_path).convert('RGB')

# Define preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Apply transform
image = preprocess(image)
image.requires_grad = True  # Enable gradient tracking

# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Forward pass
output = model(image.unsqueeze(0))  # Add batch dimension
class_idx = torch.argmax(output)
output[0, class_idx].backward()

# Generate and visualize saliency map
saliency = image.grad.abs().squeeze().permute(1, 2, 0)
plt.imshow(saliency.numpy(), cmap="hot")
plt.title("Saliency Map")
plt.axis("off")
plt.show()
