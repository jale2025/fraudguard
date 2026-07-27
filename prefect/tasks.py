from src.data_helper import FileLists, data_available, parquet_to_sql, move_file, is_parquet, validate_data_files, database_url
from prefect import task
from pathlib import Path


import os
import subprocess

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

    

@task
def dbt_build(project_dir: str = os.environ["DBT_PROJECT_DIR"], target: str = "dev") -> bool:
    """prefect task to run the dbt build command

    Args:
        project_dir (str, optional): dbt project directory. Defaults to os.environ["DBT_PROJECT_DIR"].
        target (str, optional): dbt target. Defaults to "dev".

    Raises:
        RuntimeError: raised if dbt build failed

    Returns:
        int: The return code of the dbt build command.
    """
    command = [
        "dbt",
        "build",
        "--project-dir", str(Path(project_dir).resolve()),
        "--profiles-dir", str(Path(project_dir).resolve()),
        "--target", target,
    ]

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(f"DBT BUILD OUTPUT:\n{result.stdout}")

    if result.returncode != 0:
        raise RuntimeError(
            f"dbt build failed:\n{result.stdout}"
        )

    return result.returncode == 0


# @task
# def