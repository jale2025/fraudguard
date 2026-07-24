import prefect
from prefect import flow, task


@task
def sdfs():
    d = 2


@flow
def pipeline():
    a = 2


@task
def check_condition() -> bool:
    # Deine Logik: z. B. Data Drift erkannt, neue Daten in der DB, etc.
    data_available = True 
    return data_available

@task
def process_data():
    print("Verarbeite Daten...")

@task
def run_dbt_models():
    print("Führe dbt aus...")

@flow(name="fraud_detection_pipeline")
def my_pipeline():
    # Task 1 ausführen und Rückgabewert erhalten
    should_run = check_condition()

    # Tasks 2-5 nur ausführen, wenn Task 1 True liefert
    if should_run:
        process_data()
        run_dbt_models()
        # ... weitere Tasks
    else:
        print("Bedingung nicht erfüllt – Tasks 2 bis 5 werden übersprungen.")

if __name__ == "__main__":
    my_pipeline()

