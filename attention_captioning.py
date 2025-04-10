from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# Step 1: Load your image
image_path = "sample.jpg"  # Make sure this file exists in your folder
image = Image.open(image_path).convert("RGB")

# Step 2: Load model components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Step 3: Preprocess the image and generate caption
inputs = processor(images=image, return_tensors="pt")
caption_ids = model.generate(**inputs)

# Step 4: Decode and print the caption
caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
print(f"Generated Caption with Attention: {caption}")
