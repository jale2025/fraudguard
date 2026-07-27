import prefect
from prefect import flow, task
from tasks import data_ingestion
from src.data_helper import data_available


@task
def run_dbt_models():
    print("Führe dbt aus...")

@flow(name="fraud_detection_pipeline")
def my_pipeline():

    should_run = data_available()

    if should_run:

        data_ingested = data_ingestion()

    if data_ingested:

        run_dbt_models()

if __name__ == "__main__":
    my_pipeline()

