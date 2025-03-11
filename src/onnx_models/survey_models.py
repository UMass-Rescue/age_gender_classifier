# survey_models.py

import sys
import os
import random
from age_classify_v001.model import predict as predict_age_classifier
from vit_age_classifier.model import predict as predict_vit_age_classifier

def label_to_binary(label, threshold):
    """
    Convert the age range label to a binary value based on a threshold.
    
    Args:
        label (str): Age range label (e.g., "10-19", "20-29", "more than 70").
        threshold (int): The age threshold (e.g., 20 or 30).
    
    Returns:
        bool: True if the lower bound of the age range is >= threshold, else False.
    """
    if label.lower().startswith("more"):
        # Treat "more than 70" as age 70 for threshold comparisons.
        age_lower = 70
    else:
        try:
            parts = label.split('-')
            age_lower = int(parts[0])
        except Exception:
            age_lower = 0
    return age_lower >= threshold


class SurveyModels:
    def __init__(self):
        # Hardcoded paths to the ONNX models
        self.model1_path = "age_classify_v001/v001_model.onnx"
        self.model2_path = "vit_age_classifier/vit_model.onnx"
        
        # Check if model files exist
        if not os.path.exists(self.model1_path):
            raise FileNotFoundError(f"Model file not found: {self.model1_path}")
        if not os.path.exists(self.model2_path):
            raise FileNotFoundError(f"Model file not found: {self.model2_path}")

    def predict(self, image_path):
        """
        Run (both, third to be added) age classification models on the input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary containing the predicted label and confidence for each model.
                  e.g., {"age_classify_v001": (label, confidence), "vit_age_classifier": (label, confidence)}
        """
        # Run the first model
        model1_label, model1_confidence = predict_age_classifier(image_path, self.model1_path)
        
        # Run the second model
        model2_label, model2_confidence = predict_vit_age_classifier(image_path, self.model2_path)
        
        return {
            "age_classify_v001": (model1_label, model1_confidence),
            "vit_age_classifier": (model2_label, model2_confidence)
        }

    def predict_over_under(self, age, image_path):
        """
        Poll (both, TODO add third) models and convert their age predictions into a binary over/under the specified {age} value.

        Args:
            age (int): The age threshold for the binary classification. Should be one of the following values: (3, 10, 20, 30, 40, 50, 60, 70).
            image_path (str): Path to the input image.

        Returns:
            tuple: A tuple (binary_value, confidence) where:
                - binary_value (bool) is True if "over {age}", False if "under {age}".
                - confidence (float) is the confidence score of the chosen model.

        Raises:
            ValueError: If `age` is not one of the allowed threshold values.
        """
        
        allowed_ages = {3, 10, 20, 30, 40, 50, 60, 70}
        if age not in allowed_ages:
            raise ValueError(f"Invalid age threshold: {age}. Allowed values are {sorted(allowed_ages)}.")
        
        predictions = self.predict(image_path)
        binary_results = {}
        for model_name, (label, conf) in predictions.items():
            binary_results[model_name] = (label_to_binary(label, age), conf)
        
        # Randomly choose one model's result for now
        chosen_model = random.choice(list(binary_results.keys()))
        return binary_results[chosen_model]  # Returns (binary_value, confidence)


# Command Line Execution
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python survey_models.py <image_path> <age_threshold>")
        print("Allowed age thresholds: 3, 10, 20, 30, 40, 50, 60, 70")
        sys.exit(1)
    
    image_path = sys.argv[1]

    try:
        age_threshold = int(sys.argv[2])
    except ValueError:
        print("Error: Age threshold must be an integer.")
        sys.exit(1)

    allowed_ages = {3, 10, 20, 30, 40, 50, 60, 70}
    if age_threshold not in allowed_ages:
        print(f"Error: Invalid age threshold {age_threshold}. Allowed values: {sorted(allowed_ages)}")
        sys.exit(1)

    try:
        survey = SurveyModels()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    # Full Model Predictions
    predictions = survey.predict(image_path)
    print("\nModel Predictions:")
    for model_name, (label, confidence) in predictions.items():
        print(f"{model_name}: {label} (confidence: {confidence:.2f})")
    
    # Over/Under Prediction for the specified age
    binary_result, conf_result = survey.predict_over_under(age_threshold, image_path)
    print(f"\nOver/Under {age_threshold}: {'Over' if binary_result else 'Under'} {age_threshold} (confidence: {conf_result:.2f})")
