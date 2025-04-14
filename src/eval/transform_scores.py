# from pathlib import Path
import logging
import json
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

from src.utils.common import read_db

logging.basicConfig(level=logging.INFO)
# path = Path(__file__).parent


def flatten_json_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Expand json scores from model_output."""
    scores_expanded = pd.json_normalize(df["scores"].apply(lambda x: json.loads(x)))
    df = df.join(scores_expanded).drop("scores", axis=1)
    logging.info(" Expanded JSON scores")
    return df


def main(t_stamp: str) -> pd.DataFrame:
    """Read values from DB where created_at = t_stamp,
    expand json scores into separate columns, and return a dataframe.
    """
    df = read_db(
        table_name="model_output",
        query=f"SELECT * FROM model_output where created_at = '{t_stamp}' order by created_at"
    )
    df = flatten_json_scores(df)

    # TODO: expand metrics on predicted

    return df


if __name__ == "__main__":
    main()
