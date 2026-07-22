import os
from pathlib import Path
from urllib.request import urlretrieve
from time import time

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine
from dotenv import load_dotenv

PIPE_ROOT = Path(__file__).resolve().parent # Current path of the python script
REPO_ROOT = PIPE_ROOT.parents[0] # Go one level up from the pipe root



def _database_url() -> str:
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    db_name = os.environ["DB_DATABASE"]
    print(f"Database URL the data gets ingested to: postgresql://{user}:{password}@{host}:{port}/{db_name}")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def data_ingestion(table_name: str, file_path: str) -> None:
    """This script reads a parquet file from a given url and writes it to a postgres database."""

    # Check the folder of the filepath variable exist
    # If not the missing folder will be created
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Create the sql alchemy engine object
    engine = create_engine(_database_url())

    # Read the parquet file and store it in a df
    df_transactions = pd.read_parquet(f"{file_path}")

    # Take the df columns and create an empty table with the given column schema
    df_transactions.head(n=0).to_sql(name=table_name, con=engine, if_exists="replace", index=False)

    # Open the parquet file object
    parquet_file = pq.ParquetFile(f"{file_path}")

    # Fill the database in batches by iterations
    for batch in parquet_file.iter_batches(batch_size=100000):
        start_time = time()
        batch_df = batch.to_pandas()
        batch_df.to_sql(f"{table_name}", engine, if_exists="append", index=False)
        end_time = time()
        print("Batch time: ", end_time - start_time)
    else:
        print("Finished writing to database!")


if __name__ == "__main__":
    data_ingestion(
        table_name="transactions",
        file_path=str(REPO_ROOT / "data" / "base_dataset.parquet"),
    )
