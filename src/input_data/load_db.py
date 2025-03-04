import os
from pathlib import Path
import logging
import pandas as pd
from src.utils.sqlAlchemy_manager import DBManager

logging.basicConfig(level=logging.INFO)


def read_csv_and_format(fileIn: str="age_gender.csv") -> pd.DataFrame:
    "Read labeled data set and return DataFrame"
    path = Path(__file__).parent
    file = str(path / fileIn)
    if not os.path.exists(file):
        raise FileNotFoundError(f" File not found: {file}")
    return pd.read_csv(file)


def clean_load_to_db(df: pd.DataFrame, table_name: str="age_gender_labeled") -> None:
    """Connect to DB, create/truncate table, write df to table."""
    db_uri = os.getenv("DB_CONN_STR")
    db = DBManager(db_uri, table_name)

    db.create_table_if_not_exists()
    db.truncate_table()

    df.to_sql(table_name, con=db.engine, if_exists="append", index=False)
    ct = pd.read_sql(f"SELECT COUNT(*) FROM {table_name}", con=db.engine).iloc[0, 0]
    logging.info(f" Clean loaded {ct} records into DB table {table_name}")


def main() -> None:
    df = read_csv_and_format()
    clean_load_to_db(df)


if __name__ == "__main__":
    main()
