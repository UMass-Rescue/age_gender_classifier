# survey_models.py

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List
import logging
import pandas as pd
import random
from src.onnx_models.age_classify_v001.model import predict as predict_age_classifier
from src.onnx_models.vit_age_classifier.model import predict as predict_vit_age_classifier

from src.utils.common import write_db, read_db

logging.basicConfig(level=logging.INFO)


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
        self.base_path = Path(__file__).resolve().parent
        self.model1_path = str(self.base_path / "age_classify_v001/v001_model.onnx")
        self.model2_path = str(self.base_path / "vit_age_classifier/vit_model.onnx")
        self.now = datetime.now().isoformat()

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
        
        return  {
            "model_name": ["age_classify_v001", "vit_age_classifier"],
            "scores": [
                {"label": model1_label, "confidence": str(model1_confidence)},
                {"label": model2_label, "confidence": str(model2_confidence)},
            ],
            "created_at": [self.now, self.now]
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
        
        pred = self.predict(image_path)
        binary_results = {}
        
        for n in range(len(pred["model_name"])):
            binary_results[pred["model_name"][n]] = (
                label_to_binary(
                    pred["scores"][n]["label"], age
                ),
                pred["scores"][n]["confidence"]
            )
        
        # Randomly choose one model's result for now
        chosen_model = random.choice(list(binary_results.keys()))

        return {
            "model_name": ["predict_over_under"],
            "scores": [{"chosen_model": chosen_model, "binanry_results": binary_results},
            ],
            "created_at": [self.now]
        }

    def main_predict(self, images: List, age_threshold: int=40) -> pd.DataFrame:
        """Loop list of images, for each, run prediction and write results to db."""
        counts = 0
        for img in images:
            df1 = pd.DataFrame(self.predict(img))
            df2 = pd.DataFrame(self.predict_over_under(age_threshold, img))
            dfs = [df1, df2]

            df = pd.concat(dfs, axis=0)
            df["scores"] = df["scores"].apply(lambda x: json.dumps(x))

            write_db(df, "model_output")
            counts += 1 
        logging.info(f" Successfully completed {len(df)} predictions for {counts} images.")
        return df


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
    # predictions = survey.predict(image_path)
    
    # print("\nModel Predictions:")
    # print(predictions)
    # for x in predictions.items():
    #     print(x)
    # for model_name, (label, confidence) in predictions.items():
    #     print(f"{model_name}: {label} (confidence: {confidence:.2f})")
    
    # Over/Under Prediction for the specified age
    # binary_result, conf_result = survey.predict_over_under(age_threshold, image_path)
    # chose_model = survey.predict_over_under(age_threshold, image_path)
    # print(chose_model)
    # print(f"\nOver/Under {age_threshold}: {'Over' if binary_result else 'Under'} {age_threshold} (confidence: {conf_result:.2f})")
    df = survey.main_predict([image_path], age_threshold=40)
    df2 = read_db("model_output", "SELECT * FROM model_output")
    
    logging.info("Current run", df)
    logging.info("Read DB", df2.head())
