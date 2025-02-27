from pathlib import Path
import pandas as pd


def formatter(fileIn: str="age_gender.csv", fileOut: str="age_gender_indexed.csv") -> None:
    "Read labeled data set, split pixels into array of int, write to file_indexed"
    path = Path(__file__).parent

    df = pd.read_csv(str(path / fileIn))
    df['pixels'] = df['pixels'].apply(lambda x: x.split(' '))
    df.to_csv(str(path / fileOut), index=True, header=None)


if __name__ == "__main__":
    formatter()
