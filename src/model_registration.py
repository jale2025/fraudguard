import os
import time

import mlflow
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import RandomizedSearchCV

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ["MODEL_NAME"]
DEFAULT_MODEL_ALIAS = os.environ["DEFAULT_MODEL_ALIAS"]


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
            raise RuntimeError(
                f"Model version {model_name} v{version} failed registration."
            )
        time.sleep(1)

    raise TimeoutError(
        f"Timed out waiting for {model_name} v{version} to become READY."
    )


def register_model(
    rnd_search_cv_obj: RandomizedSearchCV,
    training_rows: int,
    test_rows: int,
    recall_train: float,
    recall_test: float,
    input_schema: ModelSignature,
    input_example: pd.DataFrame,
    model_name: str = MODEL_NAME,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    timeout_seconds: int = 60,
) -> bool:
    """
    Register the best model in MLflow and promote it to production if it outperforms the current production model.

    Args:
        rnd_search_cv_obj (RandomizedSearchCV): Randomized search object containing the best estimator.
        training_rows (int): Number of rows in the training set.
        test_rows (int): Number of rows in the test set.
        recall_train (float): Recall metric for the training set.
        recall_test (float): Recall metric for the test set.
        input_schema (ModelSignature): Signature of the input data.
        input_example (pd.DataFrame): Example of the input data.
        model_name (str): Name of the model to register.
        model_alias (str): Alias for the registered model.
        mlflow_tracking_uri (str): MLflow tracking URI.
        timeout_seconds (int): Timeout in seconds

    """
    # Set the mlflow tracking uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # Print the mlflow tracking uri
    print(f"Using MLflow tracking uri: {mlflow_tracking_uri}")

    # Create the mlflow client
    client = MlflowClient()

    # Set the experiment name
    experiment = mlflow.set_experiment(experiment_name="fraud_detection")
    print(f"Set the experiment name = {experiment.name}")

    # Start running the workflow
    with mlflow.start_run(run_name="fraud_detection_model") as run:
        # Logging of workflow parameters
        mlflow.log_param("training_rows", training_rows)
        mlflow.log_param("test_rows", test_rows)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("alias", model_alias)

        # Logging of workflow metrics
        mlflow.log_metric("train_recall", recall_train)
        mlflow.log_metric("test_recall", recall_test)

        # Log the model
        mlflow_sklearn.log_model(
            rnd_search_cv_obj.best_estimator_,
            name=MODEL_NAME,
            serialization_format="skops",
            signature=input_schema,
            input_example=input_example,
        )

        # Set the model uri of the current run including the run id
        model_uri = f"runs:/{run.info.run_id}/{model_name}"

    # Print the run id and the model uri
    print(f"Logged run {run.info.run_id}")
    print(f"Registering {model_uri} as {model_name}")

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

    # Check the current recall is better than the registered recall score
    recall_prod = None

    try:
        # Try to retrieve the current production model version using the alias
        prod_model_version = client.get_model_version_by_alias(
            name=model_name, alias=model_alias
        )
        prod_run = client.get_run(str(prod_model_version.run_id))

        # Retrieve the test_recall metric logged in that run
        recall_prod = prod_run.data.metrics.get("test_recall")
        print(
            f"Current production model (v{prod_model_version.version}) "
            f"with Test Recall: {recall_prod:.4f}"
        )

    except Exception:
        # Triggered if no model currently holds the 'production' alias
        print(f"No existing production model found with alias '{model_alias}'.")

    # Evaluate if the newly trained model outperforms the current production baseline
    if recall_prod is None or recall_test > recall_prod:
        # # Ensure the destination folder exists
        # output_dir = Path("models")
        # output_dir.mkdir(parents=True, exist_ok=True)
        # skops_file_path = output_dir / "best_fraud_model.skops"

        # # Save the best pipeline/estimator as a .skops file
        # sio.dump(rnd_search_cv_obj.best_estimator_, skops_file_path)

        # Promote the new model version to production in MLflow
        client.set_registered_model_alias(
            name=model_name, alias=model_alias, version=model_version.version
        )

        prev_score_str = "None" if recall_prod is None else f"{recall_prod:.4f}"
        print(
            f"NEW BEST MODEL! Test Recall improved from "
            f"{prev_score_str} to {recall_test:.4f}.\n"
            # f"Saved artifact locally at: {skops_file_path}\n"
            f"Updated MLflow alias '{model_alias}' -> Version {model_version.version}."
        )

        return True

    else:
        print(
            f"No improvement. Current Test Recall ({recall_test:.4f}) "
            f"is not higher than Production Recall ({recall_prod:.4f}).\n"
            # f"Local .skops file and Production alias remain unchanged."
        )

    return False
