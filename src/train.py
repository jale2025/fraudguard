import pandas as pd
import polars as pl
import os
import time
from pathlib import Path
import click


DEFAULT_INPUT_PATH = Path("data/base_dataset.parquet")
DEFAULT_MODEL_NAME = "rnd_search_cv_rnd_forest_classifier"
DEFAULT_ALIAS = "production"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

SEED = 42

DB_URI = "postgresql://postgres:postgres@localhost:5433/ny_taxi"

def read_base_dataset(filename: str) -> pd.DataFrame:
    """
    Read the parquet file of the base dataset

    Args:
        filename (str): Filename of the base dataset

    Returns:
        pd.DataFrame: Dataframe of the base dataset
    """

    return pl.read_parquet(filename)


def write_dataset_in_postgresql_db(df: pd.DataFrame, db_uri: str, table_name: str) -> None:
    """
    Write the base dataset in the postgresql database

    Args:
        df (str): Dataframe containing the base dataset
        db_uri (str): Postgresql database uri
        table_name (str): Table name the dataset will be stored
    """

    # Write the dataset from the df in the postgresql database
    df.write_database(
        table_name=table_name,
        connection=db_uri,
        if_table_exists="replace",
        engine="adbc",
    )


def get_data_from_postgresql_db(db_uri: str, query: str) -> pd.DataFrame:
    """
    Read data from the postgresql database

    Args:
        db_uri (str): Postgresql database uri
        query (str): Database query to get all data

    Returns:
        pd.DataFrame: Get all data from the postgresql database 
    """

    df = pl.read_database_uri(
    query=query,
    uri=db_uri,
    engine="adbc"
    )

    return df

def wait_for_model_version(client, model_name, version, timeout_seconds):
    """Wait until the registered model version is ready to serve."""
    # MLflow registration can finish asynchronously depending on the backend.
    # Polling here keeps the local workflow predictable before the API tries
    # to resolve the production alias.
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(model_name, version)
        status = model_version.status
        if status == "READY":
            return model_version
        if status == "FAILED_REGISTRATION":
            raise click.ClickException(
                f"Model version {model_name} v{version} failed registration."
            )
        time.sleep(1)

    raise click.ClickException(
        f"Timed out waiting for {model_name} v{version} to become READY."
    )


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_INPUT_PATH,
    show_default=True,
    help="CSV file used to bootstrap the local model.",
)
@click.option(
    "--mlflow_tracking-uri",
    default=DEFAULT_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI used for logging and registration.",
)
@click.option(
    "--model-name",
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Registered model name used by the monitoring API.",
)
@click.option(
    "--alias",
    default=DEFAULT_ALIAS,
    show_default=True,
    help="Model alias that the API resolves through MLflow.",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    show_default=True,
    help="Fraction of rows reserved for the holdout evaluation split.",
)
@click.option(
    "--random-state",
    type=int,
    default=SEED,
    show_default=True,
    help="Random seed used for the train/test split.",
)
@click.option(
    "--timeout-seconds",
    type=int,
    default=60,
    show_default=True,
    help="Maximum time to wait for MLflow model registration to finish.",
)
def main(    
        input_path,
        mlflow_tracking_uri,
        model_name,
        alias,
        test_size,
        random_state,
        timeout_seconds,
    ):

    # Imports
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.metrics import recall_score
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    import os
    import time
    from pathlib import Path
    import click
    import skops.io as sio
    from sqlalchemy import text, create_engine


    # Raise an exception for the case the input path variable does not exist
    if not input_path.exists():
        raise click.ClickException(
            f"{input_path} does not exist. This repo expects the bundled reference CSV "
            "to be available before bootstrapping the local model."
        )
    
    # Set the mlflow tracking uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Print the mlflow tracking uri
    click.echo(f"Using MLflow tracking uri: {mlflow_tracking_uri}")

    # Read the parquet file of the base dataset
    df_fraud_detection = read_base_dataset(filename=input_path)

    engine = create_engine("postgresql://postgres:postgres@localhost:5433/ny_taxi")

    with engine.begin() as connection:
        connection.execute(text("DROP VIEW IF EXISTS yellow_taxi_clean"))

    # Create an empty table with the schema inferred from the DataFrame.
    # df_fraud_detection.head(0).to_sql(name="transactions", con=engine, if_exists="replace", index=False)
    # print(f"Created or replaced table 'transactions'.")

    # Write the base dataset in the postgresql database
    write_dataset_in_postgresql_db(df=df_fraud_detection, db_uri=DB_URI, table_name="transactions")

    # Get all data of the postgresql database
    df_fraud_detection_sql = get_data_from_postgresql_db(db_uri=DB_URI, query="SELECT * FROM transactions")

    # Split data in train/test data
    col_name_class = "Class"
    col_name_amount = "Amount"

    # Transform polars df into pandas df
    df_fraud_detection_sql = df_fraud_detection_sql.to_pandas()

    # Set the features and the target
    X = df_fraud_detection_sql.drop(columns=[col_name_class])
    y = df_fraud_detection_sql[col_name_class]

    # Apply the train  test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    # Define the preprocessing object
    preprocessing = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), [col_name_amount]),    
        ],
        remainder='passthrough',
    )

    # Define the pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessing', preprocessing),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1)),
        ]
    )

    # Set the parameter for the randomized search cv
    params = {
        'classifier__n_estimators': [150, 200],
        'classifier__max_depth': [10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__max_features': ['sqrt', 'log2']
    }

    # Define the object for the cross validation
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

    # Apply the randomzied search cv  in combination with the pipeline
    rnd_search_cv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=1,             
        scoring='recall',     
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        verbose=2
    )

    # Fit the model
    rnd_search_cv.fit(X_train, y_train)

    # Calculate the predictions for the train dataset
    pred_y_train = rnd_search_cv.predict(X_train)

    # Calculate the predictions for the test dataset
    pred_y_test = rnd_search_cv.predict(X_test) 

    # Calculate the recall score for the train and test dataset
    recall_train =  recall_score(y_train, pred_y_train)
    recall_test = recall_score(y_test, pred_y_test)

    # Print process information and metrics
    click.echo(
        f"Trained RandomForestClassifier on {len(X_train)} rows, holdout size {len(X_test)}."
    )
    click.echo(f"Train Recall: {np.round(recall_train, 3)}")
    click.echo(f"Test Recall: {np.round(recall_test, 3)}")

    # Create an input schema for the mlflow ui interface to see which columns and data types are expected 
    input_example = X_train.head(5).astype(float)
    input_schema = infer_signature(input_example, rnd_search_cv.predict(input_example))

    # Create the mlflow client
    client = MlflowClient()

    # Set the experiment name
    experiment = mlflow.set_experiment(experiment_name="fraud detection")
    click.echo(f"Set the experiment name = {experiment.name}")

    # Start running the workflow
    with mlflow.start_run(run_name="fraud detection model") as run:

        # Logging of workflow parameters
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("alias", alias)

        # Logging of workflow metrics
        mlflow.log_metric("train_recall", recall_train)
        mlflow.log_metric("test_recall", recall_test)

        # Log the model
        mlflow.sklearn.log_model(
            rnd_search_cv.best_estimator_,
            name="rnd_search_cv_rnd_forest_classifier",
            serialization_format="skops",
            signature=input_schema,
            input_example=input_example,
        )

        # Set the model uri of the current run including the run id
        model_uri = f"runs:/{run.info.run_id}/rnd_search_cv_rnd_forest_classifier"

    # Print the run id and the model uri
    click.echo(f"Logged run {run.info.run_id}")
    click.echo(f"Registering {model_uri} as {model_name}")

    # Register the run artifact as a named model, then point the alias used by
    # the API at the new version.

    # Register the model and all corresponding meta data and get the registration object back
    registration = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Get the model version of the current run
    model_version = wait_for_model_version(
        client=client,
        model_name=model_name,
        version=registration.version,
        timeout_seconds=timeout_seconds,
    )

    # # Add an alias to the registered model
    # client.set_registered_model_alias(
    #     name=model_name, alias=alias, version=model_version.version
    # )

    # Print the registered model (version) and the alias
    # click.echo(
    #     f"Registered {model_name} version {model_version.version} "
    #     f"and assigned alias '{alias}'."
    # )

    # # Print the alias under which you can find the model
    # click.echo(
    #     "The FastAPI service can now resolve models:/"
    #     f"{model_name}@{alias} from your local MLflow server."
    # )

    # Check the current recall is better than the registered recall score
    recall_prod = None
    
    try:
        # Try to retrieve the current production model version using the alias
        prod_model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
        prod_run = client.get_run(prod_model_version.run_id)
        
        # Retrieve the test_recall metric logged in that run
        recall_prod = prod_run.data.metrics.get("test_recall")
        click.echo(
            f"Current production model (v{prod_model_version.version}) "
            f"with Test Recall: {recall_prod:.4f}"
        )
        
    except Exception:
        # Triggered if no model currently holds the 'production' alias 
        click.echo(f"No existing production model found with alias '{alias}'.")

    # Evaluate if the newly trained model outperforms the current production baseline
    if recall_prod is None or recall_test > recall_prod:
        
        # Ensure the destination folder exists
        output_dir = Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)
        skops_file_path = output_dir / "best_fraud_model.skops"

        # Save the best pipeline/estimator as a .skops file
        sio.dump(rnd_search_cv.best_estimator_, skops_file_path)
        
        # Promote the new model version to production in MLflow
        client.set_registered_model_alias(
            name=model_name, alias=alias, version=model_version.version
        )
        
        prev_score_str = "None" if recall_prod is None else f"{recall_prod:.4f}"
        click.echo(
            f"NEW BEST MODEL! Test Recall improved from "
            f"{prev_score_str} to {recall_test:.4f}.\n"
            f"Saved artifact locally at: {skops_file_path}\n"
            f"Updated MLflow alias '{alias}' -> Version {model_version.version}."
        )
        
    else:
        click.echo(
            f"No improvement. Current Test Recall ({recall_test:.4f}) "
            f"is not higher than Production Recall ({recall_prod:.4f}).\n"
            f"Local .skops file and Production alias remain unchanged."
        )

if __name__ == "__main__":
    main()