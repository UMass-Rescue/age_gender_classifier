import matplotlib.pyplot as plt
import os, cv2 as cv, logging
import pandas as pd
from src.utils.sqlAlchemy_manager import DBManager
import numpy as np

logging.basicConfig(level=logging.INFO) # shows logs in the system

def load_data_from_db(table_name: str = "age_gender_labeled") -> pd.DataFrame:
  ''' Connect to DB, read the table '''
  db_uri = os.getenv("DB_CONN_STR")
  db = DBManager(db_uri, table_name)
  query = f"SELECT age, ethnicity, gender, pixels FROM {table_name}"
  df = pd.read_sql(query, con=db.engine)
  engine = db.engine
  db.commit_and_close()
  return df, engine

def is_valid_image(pixel_string):
   ''' checks if pixel string exactly contains 48*48 = 2304 values '''
   pixel= pixel_string.strip().split()
   return len(pixel) == 48*48

def is_damaged(image, threshold_low = 10, threshold_high = 255, min_variance = 10):
   """ 
   Detects damaged images based on pixel intensity distribution.
   - threshold_low: If pixels are below this, the image may be too dark.
   - threshold_high: If pixels are above this, the image may be too bright.
   - min_variance: If the variance is too low, image may lack details
   """
   if np.mean(image) < threshold_low or np.mean(image) > threshold_high:
      return True # likely corrupted
   # Check if the image lacks details (variance too low)
   if np.mean(image) < min_variance:
      return True # likely corrupted
   return False

def enhance_image(image):
    '''kernel, sharpening, denoising, and bilateral filtering techniques for image preprocessing'''
    kernel = np.array([[0, -0.5, 0], 
                       [-0.5, 3, -0.5], 
                       [0, -0.5, 0]])  # Sharpening
    sharpened = cv.filter2D(image, -1, kernel)
    smoothed = cv.bilateralFilter(sharpened, d=3, sigmaColor=55, sigmaSpace=55)
    denoised = cv.fastNlMeansDenoising(smoothed, h=5, templateWindowSize=7, searchWindowSize=21)
    return denoised

def show_images(df):
    # Plot settings
    plt.figure(figsize=(20, 30))
    plt.subplots_adjust(wspace=0.6, hspace=1)

    for i in range(10):
        index = np.random.randint(0, len(df))
        original = df["pixels_array"].iloc[index]
        enhanced = enhance_image(original)

        # Original image (left)
        plt.subplot(10, 2, 2 * i + 1)
        plt.imshow(original, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f"Original\nAge: {df['age'].iloc[index]}, Ethnicity: {['White', 'Black', 'Asian', 'Indian', 'Hispanic'][df['ethnicity'].iloc[index]]}, Gender: {['Male', 'Female'][df['gender'].iloc[index]]}",
            loc="left", color="red", fontsize=9, pad=15
        )

        # Enhanced image (right)
        plt.subplot(10, 2, 2 * i + 2)
        plt.imshow(enhanced, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("Enhanced", loc="left", color="blue", fontsize=9, pad=15)

    plt.show()

'''Cleaning the dataset'''
df, engine = load_data_from_db()
df = df[df["pixels"].apply(is_valid_image)].copy()
df["pixels_array"] = df["pixels"].apply(lambda x: np.array([int(p) for p in x.split()], dtype=np.uint8).reshape(48, 48))
df = df[~df["pixels_array"].apply(is_damaged)]
df.drop(columns=["pixels_array"]).to_sql("cleaned_age_gender_labeled", engine, index=False, if_exists="replace")
