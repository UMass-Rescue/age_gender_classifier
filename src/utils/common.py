import os
from typing import Optional
import logging
import pandas as pd
from src.utils.sqlAlchemy_manager import DBManager

logging.basicConfig(level=logging.INFO)
db_uri = os.getenv("DB_CONN_STR")


def read_db(
        table_name: str="age_gender_labeled",
        query: Optional[str]=None
    ) -> pd.DataFrame:
    """Read records from custom query into pandas DataFrame."""
    if query is None:
        logging.warning("No query provided. Returning None.")
        return None

    db = DBManager(db_uri, table_name)
    df = pd.read_sql(query, con=db.engine)
    db.commit_and_close()

    logging.info(f" Retrieved {len(df)} records from DB table {table_name}")
    return df


def write_db(
        df: pd.DataFrame,
        table_name: str="age_gender_labeled"
    ) -> None:
    """Write pandas DataFrame into DB table."""

    db = DBManager(db_uri, table_name)
    df.to_sql(table_name, con=db.engine)
    db.commit_and_close()

    logging.info(f" Wrote {len(df)} records from DB table {table_name}")
