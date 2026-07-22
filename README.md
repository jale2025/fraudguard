# fraudguard

## Setup

You will need **Docker Desktop** installed and running on your machine. If you do not have it installed, please follow the [installation instructions](https://docs.docker.com/get-docker/).

Before running the project, copy `.env.example` to `.env` from the repository root.

Git BASH / Linux / macOS:
```BASH / Linux / macOS
cp .env.example .env
```

PowerShell:
```Powershell
Copy-Item .env.example .env
```

## Data preparation 

For the pre-preparations we downloaded the data from [kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/) and separated it in a first step to be able to simulate a full MLOps lifecycle.
The original dataset was split into 90% base dataset a 9:1 ratio. The base:dataset.parquet contains most of the dataand is later used for training the baseline model. The remaining 10% of the data was split in a 1:1 ratio in api streaming data (api_data.parquet) and retraining streaming data (retrain_data.parquet). These files are visible in the data and data/external folders. The split was done using scikit-learn.train_test_split(SET_SEED=42) and test_size=(0.1, 0.5).
