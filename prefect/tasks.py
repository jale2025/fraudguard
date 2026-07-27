from src.data_helper import FileLists, data_available, parquet_to_sql, move_file, is_parquet, validate_data_files, database_url
from prefect import task
from pathlib import Path
import os

@task
def data_ingestion(dir_path: str | Path = os.environ["DATA_DIR_INCOMING"]) -> bool:
    """
    Validate all incoming files and write the valid parquet files in the database.

    Args:
        dir_path (str | Path, optional): Used directory to validate all containing files. Defaults to os.environ["DATA_DIR_INCOMING"].

    Returns:
        bool: True if at least one parquet file was written in the database.
    """

    # Create a list of all valid parque lists
    file_lists = validate_data_files(dir_path)

    # Create a counter for the number of written parquet files
    counter = 0

    # Iterate through all valid parquet files
    for parquet_file in file_lists.valid_list:

        # Write the parquet file in te database
        parquet_to_sql(file_path=parquet_file)

        # Increase the counter
        counter+=1

        # Move the valid parquet file to the proecessed folder
        move_file(source_path=parquet_file, destination_dir=os.environ["DATA_DIR_PROCESSED"])

    # Iterate through all invalid parquet files
    for parquet_file in file_lists.invalid_list:
    
            # Move the valid parquet file to the quarantine folder
            move_file(source_path=parquet_file, destination_dir=os.environ["DATA_DIR_QUARANTINE"])

    return True if counter > 0 else False

    

