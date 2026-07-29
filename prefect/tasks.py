import os
import subprocess
from pathlib import Path

from prefect import task
from src.data_helper import (
    move_file,
    parquet_to_sql,
    validate_data_files,
)
from src.model_registration import register_model
from src.train import get_training_data, train_model

DBT_SCHEMA = os.environ["DBT_SCHEMA"]
MODEL_NAME = os.environ["MODEL_NAME"]
DEFAULT_MODEL_ALIAS = os.environ["DEFAULT_MODEL_ALIAS"]
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
SEED = 42


@task
def data_ingestion(dir_path: str | Path) -> bool:
    """
    Validate all incoming files and write the valid parquet files in the database.

    Args:
        dir_path (str | Path): Used directory to validate all containing files.

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
        counter += 1

        # Move the valid parquet file to the proecessed folder
        move_file(
            source_path=parquet_file, destination_dir=os.environ["DATA_DIR_PROCESSED"]
        )

    # Iterate through all invalid parquet files
    for parquet_file in file_lists.invalid_list:
        # Move the valid parquet file to the quarantine folder
        move_file(
            source_path=parquet_file, destination_dir=os.environ["DATA_DIR_QUARANTINE"]
        )

    return True if counter > 0 else False


@task
def dbt_build(project_dir: str, target: str = "dev") -> bool:
    """
    Prefect task to run the dbt build command.

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
        "--project-dir",
        str(Path(project_dir).resolve()),
        "--profiles-dir",
        str(Path(project_dir).resolve()),
        "--target",
        target,
    ]

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(f"DBT BUILD OUTPUT:\n{result.stdout}")

    if result.returncode != 0:
        raise RuntimeError(f"dbt build failed:\n{result.stdout}")

    return result.returncode == 0


@task
def get_data_train_model(
    db_table_name=DBT_SCHEMA,
    target_label: str = "class",
    test_size: float = 0.2,
    random_state: int = SEED,
    num_col_to_scale: str = "amount",
) -> dict:
    """
    Fetch training data and train the model.

    Args:
        db_table_name (str): Database table name to query training data from.
        target_label (str): Name of the target column.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Seed for random operations.
        num_col_to_scale (str): Numeric column name to scale.

    Returns:
        dict: Dictionary containing training results and metrics.

    """
    df_train_data = get_training_data(db_table_name=db_table_name)
    train_result = train_model(
        df_training_data=df_train_data,
        target_label=target_label,
        test_size=test_size,
        random_state=random_state,
        num_col_to_scale=num_col_to_scale,
    )
    return train_result


@task
def model_registration(
    train_result_dict: dict,
    model_name: str = MODEL_NAME,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    timeout_seconds: int = 60,
) -> None:
    """
    Register the trained model in MLflow.

    Args:
        train_result_dict (dict): Dictionary produced by the training task.
        model_name (str): Name of the MLflow model to register.
        model_alias (str): Alias to assign to the registered model.
        mlflow_tracking_uri (str): MLflow tracking server URI.
        timeout_seconds (int): Timeout for registration operations.

    """
    register_model(
        rnd_search_cv_obj=train_result_dict["model"],
        training_rows=train_result_dict["training_rows"],
        test_rows=train_result_dict["test_rows"],
        recall_train=train_result_dict["recall_train"],
        recall_test=train_result_dict["recall_test"],
        input_schema=train_result_dict["input_schema"],
        input_example=train_result_dict["input_example"],
        model_name=model_name,
        model_alias=model_alias,
        mlflow_tracking_uri=mlflow_tracking_uri,
        timeout_seconds=timeout_seconds,
    )
