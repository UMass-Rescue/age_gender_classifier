import onnxruntime as ort
import numpy as np
import cv2 as cv
from PIL import Image
from src.utils.preprocess_and_clean import enhance_image

# Preprocessing function
def preprocess_image(image_path, target_size=224):
    """
    Preprocess a single image for ONNX inference:
      - Load and convert to grayscale
      - Enhance using your custom method (expects 48x48 input)
      - Resize to model input size (e.g., 224x224)
      - Normalize to [0,1] and convert to tensor format (1, 1, 224, 224)
    """
    # Load and convert to grayscale (already 48x48 expected from cleaning)
    image = Image.open(image_path).convert("L")
    image_np = np.array(image, dtype=np.uint8)

    # Enhance
    enhanced = enhance_image(image_np)

    # Resize to model input size
    resized = cv.resize(enhanced, (target_size, target_size))

    # Normalize and convert to tensor
    final = resized.astype(np.float32) / 255.0
    final = np.expand_dims(final, axis=0)  # Channel dim: (1, 224, 224)
    final = np.expand_dims(final, axis=0)  # Batch dim: (1, 1, 224, 224)

    return final

# Postprocessing function
def postprocess_output(logits):
    """
    Apply softmax to the raw logits and return predicted classes for the batch.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    predicted_classes = np.argmax(probabilities, axis=1)
    return predicted_classes, probabilities

# Mapping from classification ids to age ranges (based on fairface config)
id2label = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "more than 70"
}

# Run inference
def predict(image_path, onnx_path):
    """
    Load ONNX model and predict age group from an image.

    Returns:
        (label: str, confidence: float)
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_tensor = preprocess_image(image_path)
    outputs = session.run(["logits"], {"pixel_values": input_tensor})[0]
    predicted_classes, probabilities = postprocess_output(outputs)
    predicted_class = predicted_classes[0]
    predicted_label = id2label.get(predicted_class, "Unknown")
    confidence = probabilities[0][predicted_class]
    return predicted_label, confidence
