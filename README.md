# FraudGuard

We developed FraudGuard during our Capstone Project of the AI Engineering Bootcamp at neuefische / Spiced academy. FraudGuard is an end-to-end MLOps application for credit card fraud detection. It serves model predictions through a FastAPI service, stores incoming transactions in PostgreSQL, monitors the system and can trigger model retraining when data drift is detected. The project uses the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Our repository contains prepared subsets for initial training and API demonstrations.

## Features

- Fraud predictions for single transactions and Parquet files
- Automated data processing and model training with Prefect and dbt
- Model tracking and registry with MLflow
- Data drift detection with Evidently
- Automatic retraining when drift is detected
- API and model monitoring with Prometheus and Grafana
- Unit tests, integration tests and code checks with GitHub Actions
- Simple browser-based demo interface

## Setup

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Docker Compose

Make sure Docker Desktop is running before starting the project.

### 1. Clone the repository

```bash
git clone https://github.com/jale2025/fraudguard.git
cd fraudguard
```

### 2. Create the environment file

Git Bash, Linux or macOS:

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

The default values are prepared for the local Docker setup and normally do not need to be changed.

### 3. Start the application

```bash
docker compose up --build
```

On the first start, FraudGuard initializes the databases, processes the included base dataset, trains the initial model and registers it in MLflow. This can take a few minutes.

The application is ready when the following URL returns `{"status":"ready"}`:

```text
http://localhost:8000/ready
```

If `/ready` returns HTTP 503 during startup, the initial model is not available yet. Wait for the first Prefect pipeline run to finish and check the worker logs if necessary.

## Services

| Service | URL | Purpose |
| --- | --- | --- |
| Demo interface | http://localhost:8000/demo | Upload a Parquet file and run fraud predictions |
| API documentation | http://localhost:8000/docs | Explore and test the API endpoints |
| MLflow | http://localhost:5000 | View experiments, metrics and registered models |
| Prefect | http://localhost:4200 | View and monitor pipeline runs |
| Grafana | http://localhost:3000 | View service, model and drift dashboards |
| Prometheus | http://localhost:9090 | Inspect collected metrics |

Grafana uses its default login (`admin` / `admin`) on the first start.

## Using FraudGuard

The easiest way to try the project is the [demo interface](http://localhost:8000/demo). Select whether the uploaded data contains known labels, upload a valid Parquet file and start the analysis.

Predictions are saved in PostgreSQL. After enough new transactions have arrived, Evidently compares them with the training data. If data drift is detected, the Prefect retraining pipeline is triggered automatically.

For direct API usage and all available endpoints, open the [FastAPI documentation](http://localhost:8000/docs).

## Useful commands

Run the services in the background:

```bash
docker compose up --build -d
```

Show running services:

```bash
docker compose ps
```

Follow the pipeline logs:

```bash
docker compose logs -f prefect-worker
```

Stop the application:

```bash
docker compose down
```

Stop the application and remove all volumes:

```bash
docker compose down -v
```