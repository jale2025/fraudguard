import os
from time import time

import pandas as pd
from data_model import (
    TransactionClassificationKnownLabel,
    TransactionClassificationUnknownLabel,
)
from redis_helper import redis_client
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError, ProgrammingError, SQLAlchemyError

DB_URI = os.getenv("DB_URI")

QUEUE_COUNTERS = {
    "predictions": "not_labeled_queue",
    "labeled_predictions_queue": "labeled_queue",
}


def _increment_queue_counters(table_name: str, rows: int) -> None:
    """
    Record how many rows arrived on a queue.

    Args:
        table_name (str): Target table the rows were written to.
        rows (int): Number of rows in this batch.

    """
    counter_name = QUEUE_COUNTERS.get(table_name)

    if counter_name is None:
        return

    client = redis_client()
    client.incr(name=counter_name, amount=rows)
    client.incr(name=f"{counter_name}_length", amount=rows)


def forward_to_database(
    table_name: str,
    data: pd.DataFrame
    | TransactionClassificationKnownLabel
    | TransactionClassificationUnknownLabel,
):
    """
    Forward data to the database in batches.

    Args:
        table_name: Name of the target database table.
        data: DataFrame or transaction classification object to insert.

    Raises:
        IntegrityError: On duplicate keys or constraint violations.
        ProgrammingError: On schema mismatches.
        SQLAlchemyError: On general database errors.

    """
    # Create the sql alchemy engine object
    engine = create_engine(DB_URI)

    if isinstance(
        data,
        (TransactionClassificationKnownLabel, TransactionClassificationUnknownLabel),
    ):
        print(
            "DEBUG: is instance TransactionClassificationKnownLabel or TransactionClassificationUnknownLabel"
        )
        data = pd.DataFrame([data.model_dump()]).copy()

    batch_size = 100000
    total_rows = len(data)

    if "target_class" in data.columns:
        print("DEBUG: column targte_class does exist")
        data.rename(columns={"target_class": "class"}, inplace=True)

    for batch_idx, start_row in enumerate(range(0, total_rows, batch_size)):
        start_time = time()

        # Batch als Slice aus dem DataFrame extrahieren
        batch_df = data.iloc[start_row : start_row + batch_size]

        try:
            batch_df.to_sql(
                f"{table_name}", engine, if_exists="append", index=False, schema="raw"
            )

            # Count the rows of *this* batch. Incrementing by total_rows here would
            # multiply the count by the number of batches, so a 250k-row upload
            # reported 750k rows and triggered the drift report far too eagerly.
            _increment_queue_counters(table_name, len(batch_df))

        except IntegrityError as e:
            # Triggered by duplicate keys or NOT NULL constraint violations
            print(
                f"Integrity Error during batch insertion (Batch {batch_idx}): {e.orig}"
            )
            raise e
        except ProgrammingError as e:
            # Triggered by schema mismatches (e.g., missing column in SQL table)
            print(
                f"Schema Error (e.g., column does not exist in target SQL table): {e.orig}"
            )
            raise e
        except SQLAlchemyError as e:
            # General database errors (e.g., connection lost)
            print(f"Database Error in batch {batch_idx}: {e}")
            raise e
        except Exception as e:
            # Unexpected Python exceptions
            print(f"Unexpected Error in batch {batch_idx}: {e}")
            raise e

        end_time = time()
        print(f"Batch {batch_idx} execution time: {end_time - start_time:.2f} seconds")
    print("Finished writing to database!")
