import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set device to CPU
device = torch.device("cpu")

# Load and preprocess image
def load_image(img_path, max_size=256, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max_size:
        size = max(max(image.size), max_size)
        in_transform = transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    elif shape:
        in_transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Load images
content_image = load_image("content.jpeg", max_size=256)
style_image = load_image("style.jpg", shape=content_image.shape[-2:])

# Display helper
def im_convert(tensor):
    image = tensor.clone().detach().squeeze(0)
    image = image.mul(torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    image = image.add(torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0)

# Load pretrained VGG19 model
from torchvision.models import vgg19, VGG19_Weights
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

# Freeze model parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Define layers to extract features from
content_layers = ['21']  # conv4_2
style_layers = ['0', '5', '10']  # conv1_1, conv2_1, conv3_1

# Create model to extract features
def get_features(image, model, layers=None):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# Gram matrix for style representation
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t()) / (d * h * w)

# Get features for style and content
content_features = get_features(content_image, vgg, content_layers)
style_features = get_features(style_image, vgg, style_layers)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Initialize target image
target = content_image.clone().requires_grad_(True)

# Style weights (can be tuned)
style_weights = {'0': 1.0, '5': 0.75, '10': 0.2}
content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training loop
steps = 300
for step in range(steps):
    target_features = get_features(target, vgg, style_layers + content_layers)
    
    # Content loss
    content_loss = torch.mean((target_features['21'] - content_features['21']) ** 2)
    
    # Style loss
    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total loss: {total_loss.item():.4f}")

# Display result
plt.imshow(im_convert(target))
plt.title("Stylized Image")
plt.axis("off")
plt.show()

