import os
 
from prefect import flow, task, get_run_logger
from tasks import data_ingestion, dbt_build
from src.data_helper import data_available

@flow(name="fraud_detection_pipeline")
def my_pipeline():
    logger = get_run_logger()

    # Check if there is data available to process
    should_run = data_available()
    logger.info(f"Data available: {should_run}")

    # If there is data available, run the data ingestion task
    if should_run:
        data_ingested = data_ingestion()
        logger.info(f"Data ingested: {data_ingested}")

    # If data ingestion was successful, run the dbt build task
    if data_ingested:
        dbt_ran = dbt_build(project_dir = os.environ["DBT_PROJECT_DIR"], target = "dev")
        logger.info(f"DBT build successful: {dbt_ran}")

    # If dbt build was successful, load the the training dataset and train the model
    if dbt_return_code:
        print("DBT build was successful. Proceeding to load the training dataset and train the model.")
        
        
if __name__ == "__main__":
    my_pipeline()

