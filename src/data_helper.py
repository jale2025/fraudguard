import os
from pathlib import Path
import shutil
from dataclasses import dataclass, field
from urllib.request import urlretrieve
from time import time
import pyarrow.parquet as pq
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, ProgrammingError
import pandas as pd

PIPE_ROOT = Path(__file__).resolve().parent # Current path of the python script
REPO_ROOT = PIPE_ROOT.parents[0] # Go one level up from the pipe root


def move_file(source_path: str | Path, destination_dir: str | Path) -> None:
    """Move a file from source_path to destination_dir with a timestamp prefix/suffix.
    
    Args:
        source_path (str | Path): Source path of the file.
        destination_dir (str | Path): Destination directory.
    """
    src = Path(source_path)
    dst_dir = Path(destination_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    current_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")

    new_filename = f"{src.stem}_{current_timestamp}{src.suffix}"

    destination_file_path = dst_dir / new_filename

    shutil.move(src, destination_file_path)


def data_available(dir_path: str | Path) -> bool: 
    """
    This function checks a file does exist in the corresponding folder.

    Args:
        dir_path (str | Path, optional): Used directory to check new files. Defaults to os.environ["DATA_DIR_INCOMING"].

    Returns:
        bool: A file does exist (true) or not (false).
    """

    # Create a path by the string
    dir_path = Path(dir_path)

    # Check the pathn does exist and it's a path
    if not dir_path.exists() or not dir_path.is_dir():
        return False

    # Check a file does exist in the folder
    return any(item.is_file() for item in dir_path.iterdir())

@dataclass
class FileLists:
    valid_list: list[str] = field(default_factory=list)
    invalid_list: list[str] = field(default_factory=list)

def validate_data_files(dir_path: str | Path) -> FileLists:
    """
    This function validates all files of the corresponding directory.

    Args:
        dir_path (str | Path, optional): Used directory to validate all containing files. Defaults to os.environ["DATA_DIR_INCOMING"].

    Returns:
        FileLists: A dataclass object contains the list of valid/invalid files.
    """

    # Create a path by the string
    dir_path = Path(dir_path)

    file_lists = FileLists()

    # Iterate through all data files
    for item in dir_path.iterdir():

        if is_parquet(item):
            file_lists.valid_list.append(str(item))
        else:
            file_lists.invalid_list.append(str(item))

    return file_lists
        


def is_parquet(file_path: str | Path) -> bool:
    """
    Check a file is a parquet file or not.

    Args:
        file_path (str | Path): String or Path of the file.

    Returns:
        bool: Returns True/False if the file is a parquet file or not.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".parquet":
        return True
    else:
        return False


def database_url() -> str:
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    db_name = os.environ["DB_DATABASE"]
    print(f"Database URL the data gets ingested to: postgresql://{user}:{password}@{host}:{port}/{db_name}")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def parquet_to_sql(file_path: str, table_name: str="transactions") -> None:
    """This script reads a parquet file from a given url and writes it to a postgres database."""

    # Check the folder of the filepath variable exist
    # If not the missing folder will be created
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Create the sql alchemy engine object
    engine = create_engine(database_url())

    # Read the parquet file and store it in a df
    # df_transactions = pd.read_parquet(f"{file_path}")

    # Take the df columns and create an empty table with the given column schema
    # df_transactions.head(n=0).to_sql(name=table_name, con=engine, if_exists="append", index=False)

    # Open the parquet file object
    parquet_file = pq.ParquetFile(f"{file_path}")

    # Fill the database in batches by iterations
    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=100000)):
        start_time = time()
        batch_df = batch.to_pandas()
        
        try:
            batch_df.to_sql(f"{table_name}", engine, if_exists="append", index=False)
        except IntegrityError as e:
            # Triggered by duplicate keys or NOT NULL constraint violations
            print(f"Integrity Error during batch insertion (Batch {batch_idx}): {e.orig}")
            raise e
        except ProgrammingError as e:
            # Triggered by schema mismatches (e.g., missing column in SQL table)
            print(f"Schema Error (e.g., column does not exist in target SQL table): {e.orig}")
            raise e
        except SQLAlchemyError as e:
            # General database errors (e.g., connection lost)
            print(f"Database Error in batch {batch_idx}: {e}")
            raise e
        except Exception as e:
            # Unexpected Python exceptions
            print(f"Unexpected Error in batch {batch_idx}: {e}")
            raise e

        end_time = time()
        print(f"Batch {batch_idx} execution time: {end_time - start_time:.2f} seconds")
    else:
        print("Finished writing to database!")


if __name__ == "__main__":
    parquet_to_sql(
        table_name="transactions",
        file_path=str(REPO_ROOT / "data" / "base_dataset.parquet"),
    )










