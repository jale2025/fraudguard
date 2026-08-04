import os

from tasks import data_ingestion, dbt_build, get_data_train_model, model_registration

from prefect import flow, get_run_logger
from src.data_helper import data_available


@flow(name="fraud_detection_pipeline", timeout_seconds=300)
def fraudguard_pipeline() -> None:
    """Execute the complete fraud detection pipeline triggered by a cron job."""
    logger = get_run_logger()

    # Initialize the bool variables
    data_ingested = False
    dbt_success = False

    # Check if there is data available to process
    should_run = data_available(dir_path=os.environ["DATA_DIR_INCOMING"])
    logger.info(f"Files available: {should_run}")

    # If there is data available, run the data ingestion task
    if should_run:
        data_ingested = data_ingestion(dir_path=os.environ["DATA_DIR_INCOMING"])
        logger.info(
            f"Files valid for ingestion have been found and ingested into database: {data_ingested}"
        )

    # If data ingestion has taken place, run the dbt build task
    if data_ingested:
        dbt_success = dbt_build(project_dir=os.environ["DBT_PROJECT_DIR"], target="dev")
        logger.info(f"DBT build successful: {dbt_success}")

    # If dbt build was successful, load the the training dataset and train the model
    if dbt_success:
        train_result_dict = get_data_train_model()
        model_registration(train_result_dict)


if __name__ == "__main__":
    fraudguard_pipeline()
    fraudguard_pipeline.serve(
        name="fraud_detection_pipeline_hourly_serve",
        cron="0 1 * * *",
    )
