from pathlib import Path
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.common import read_db

logging.basicConfig(level=logging.INFO)


def age_histogram(df: pd.DataFrame, saveFile: str="imgs/age_distribution.png") -> None:
    """"""
    sns.histplot(data=df,x='age',kde=True)
    path = Path(__file__).parent / saveFile
    plt.savefig(path)
    logging.info(f" Saved histogram to {saveFile}")


def main():
    """"""
    df = read_db(table_name="age_gender_labeled", query="SELECT * FROM age_gender_labeled")
    age_histogram(df)
    

if __name__ == "__main__":
    main()
