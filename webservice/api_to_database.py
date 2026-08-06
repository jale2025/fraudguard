
import os
from time import time

import pandas as pd
from data_model import (
    TransactionClassificationKnownLabel,
    TransactionClassificationUnknownLabel,
)
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError, ProgrammingError, SQLAlchemyError

DB_URI = os.getenv("DB_URI")


def forward_to_database(table_name: str, data: pd.DataFrame | TransactionClassificationKnownLabel | TransactionClassificationUnknownLabel):

    # Create the sql alchemy engine object
    engine = create_engine(DB_URI)

    if isinstance(data, (TransactionClassificationKnownLabel, TransactionClassificationUnknownLabel)):
        print("DEBUG: is instance TransactionClassificationKnownLabel or TransactionClassificationUnknownLabel")
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


