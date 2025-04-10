import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load a pretrained image captioning model (BLIP)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and preprocess an image
image_path = "sample.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Generate a caption
caption_ids = model.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)

# Print the generated caption
print(f"Generated Caption: {caption}")