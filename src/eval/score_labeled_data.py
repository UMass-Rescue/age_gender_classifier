from pathlib import Path
from typing import List
import logging
import json
import pandas as pd

from src.utils.common import read_db
from src.onnx_models.survey_models import SurveyModels

logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent


# TODO: ======= Juhi's work =======
def preprocess_true_pixels(pixStr: str) -> List:
    """Preprocess the true pixels from the json blob."""
    
    # put code here
    
    img = ["preprocessing here"]
    return img
# =================================


def main(mod_size: int=20) -> None:
    """Read subset of true data from database, preprocess, run inference against our models,
    and return a dataframe. Labeled data set must exist in the database.
    """
    df = read_db(
        table_name="age_gender_labeled",
        query=f"SELECT id, age, pixels FROM age_gender_labeled where id % {mod_size} = 0 order by age"
    )
    
    # =================
    # TODO @Juhi: call pre-process on labeled data, NOT file names
    # TODO # @Jake: re-org SurveyModels to handle list of png files or list of pixel arrays
    # =================
    df["pixels"] = df["pixels"].apply(preprocess_true_pixels)
    imgs = list(df["pixels"])
    ids = list(df["ids"])
    logging.info(" About to run inference on true label subset...")

    sm = SurveyModels()
    df = sm.main_predict(imgs, age_threshold=None, ids=ids)

    logging.info(" Completed scoring true labels, returning dataframe of expanded results")    
    

if __name__ == "__main__":
    main()
