from transformers import pipeline
from PIL import Image
import requests

# Option A: Load an image from a local file
image = Image.open("test_images/lebron.jpg")

# Create an image classification pipeline
classifier = pipeline(
    "image-classification",
    model="nateraw/vit-age-classifier"
)

# Run inference
predictions = classifier(image)

print(predictions)
