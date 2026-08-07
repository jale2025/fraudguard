import os

import pandas as pd
import polars as pl
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Read environment variables
DB_URI = os.environ["DB_URI"]
DBT_SCHEMA = os.environ["DBT_SCHEMA"]

SEED = 42


def get_training_data() -> pd.DataFrame:
    """
    Load training data from PostgreSQL and return it as a pandas DataFrame.

    Returns:
        pd.DataFrame: Training data.

    """
    # Get all data of the postgresql database
    df_fraud_detection_sql = pl.read_database_uri(
            query="SELECT * FROM dbt_dev_data_science.fct_training_data ORDER BY elapsed_sec, pc_1",
            uri=DB_URI,
            engine="adbc",
        )
    # return pandas df
    return df_fraud_detection_sql.to_pandas()


def train_model(
    df_training_data: pd.DataFrame,
    target_label: str = "class",
    test_size: float = 0.2,
    random_state: int = SEED,
    num_col_to_scale: str = "amount",
) -> dict:
    """
    Train and evaluate a fraud detection model.

    Args:
        df_training_data (pd.DataFrame): Input training data.
        target_label (str, optional): Name of the target column. Defaults to "class".
        test_size (float, optional): Fraction of data reserved for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to SEED.
        num_col_to_scale (str, optional): Numeric column to standardize. Defaults to "amount".

    Returns:
        dict: Trained model, recall metrics, dataset sizes, and MLflow signature example.

    """
    # Set the features and the target
    x = df_training_data.drop(columns=[target_label])
    y = df_training_data[target_label]

    # Apply the train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Define the preprocessing object
    preprocessing = ColumnTransformer(
        transformers=[
            ("num_scaler", StandardScaler(), [num_col_to_scale]),
        ],
        remainder="passthrough",
    )

    # Define the pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Set the parameter for the randomized search cv
    params = {
        "classifier__n_estimators": [150, 200],
        "classifier__max_depth": [10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__max_features": ["sqrt", "log2"],
    }

    # Define the object for the cross validation
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

    # Apply the randomized search cv in combination with the pipeline
    rnd_search_cv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=1,
        scoring="recall",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=2,
    )

    # Fit the model
    rnd_search_cv.fit(x_train, y_train)

    # Calculate the predictions for the train dataset
    pred_y_train = rnd_search_cv.predict(x_train)

    # Calculate the predictions for the test dataset
    pred_y_test = rnd_search_cv.predict(x_test)

    # Calculate the recall score for the train and test dataset
    recall_train = float(recall_score(y_train, pred_y_train))

    recall_test = float(recall_score(y_test, pred_y_test))

    # Print process information and metrics
    print(
        f"Trained RandomForestClassifier on {len(x_train)} rows, holdout size {len(x_test)}."
    )
    print(f"Train Recall: {recall_train:.3f}")
    print(f"Test Recall: {recall_test:.3f}")

    # Create an input schema for the mlflow ui interface to see which columns and data types are expected
    input_example = x_train.head(5).astype(float)
    input_schema = infer_signature(input_example, rnd_search_cv.predict(input_example))

    return_vars = {
        "model": rnd_search_cv,
        "recall_train": recall_train,
        "recall_test": recall_test,
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "input_schema": input_schema,
        "input_example": input_example,
    }

    return return_vars
