import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 1. Load the processor and the model
processor = AutoImageProcessor.from_pretrained("cledoux42/Age_Classify_v001")
model = AutoModelForImageClassification.from_pretrained("cledoux42/Age_Classify_v001")

# 2. Load the image
image = Image.open("test_images/lebron.jpg")

# 3. Preprocess the image
inputs = processor(image, return_tensors="pt")

# 4. Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 5. Find the predicted class
predicted_class_idx = logits.argmax(dim=-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

print(f"Predicted label: {predicted_label}")
