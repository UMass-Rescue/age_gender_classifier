import onnxruntime as ort
import numpy as np
from PIL import Image

# Preprocessing function
def preprocess_image(image_path, target_size=224):
    """
    Preprocess the input image:
      - Load and convert to RGB
      - Resize to target dimensions (default 224x224)
      - Convert to a numpy array and scale pixel values to [0, 1]
      - Normalize using mean and std of [0.5, 0.5, 0.5] (as float32)
      - Rearrange the array to channel-first (C, H, W) and add a batch dimension.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_size, target_size))
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    image_np = (image_np - mean) / std
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

# Postprocessing function
def postprocess_output(logits):
    """
    Apply softmax to the raw logits and return predicted classes for the batch.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    predicted_classes = np.argmax(probabilities, axis=1)
    return predicted_classes, probabilities

# Mapping from classification ids to age ranges
id2label = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "more than 70",
}


# Run inference
def predict(image_path, onnx_path):
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_tensor = preprocess_image(image_path)
    outputs = session.run(["logits"], {"pixel_values": input_tensor})[0]
    predicted_classes, probabilities = postprocess_output(outputs)
    predicted_class = predicted_classes[0]
    predicted_label = id2label.get(predicted_class, "Unknown")
    confidence = probabilities[0][predicted_class]
    return predicted_label, confidence
