import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load dataset
df = pd.read_csv("../../input_data/age_gender_indexed.csv", header=None)

df.columns = ["Index", "Age", "Ethnicity", "Gender", "Image name", "Pixels"]

def preprocess_image(pixel_str):
    try:
        # Ensure pixel_str is properly formatted
        pixel_str = pixel_str.strip("[]").replace("'", "")  # Remove quotes if any
        pixel_values = np.fromstring(pixel_str, sep=",", dtype=np.uint8)

        # Ensure correct image size (48x48 = 2304 pixels)
        if pixel_values.size != 48 * 48:
            return None

        # Reshape into 48x48 grayscale image
        image = pixel_values.reshape(48, 48)

        # Apply Bilateral Filtering for clarity while preserving edges
        image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

        # Normalize pixel values (0 to 1)
        image = image / 255.0
        
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def show_image(index):
    image = preprocess_image(df.iloc[index]["Pixels"])
    if image is None:
        print(f"Skipping index {index}: Unexpected pixel size")
        return
    
    plt.imshow(image, cmap="gray")
    plt.title(f"Age: {df.iloc[index]['Age']}, Gender: {df.iloc[index]['Gender']}")
    plt.axis("off")
    plt.show()

# Show first 5 images
for i in range(5):
    show_image(i)