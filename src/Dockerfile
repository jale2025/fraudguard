# Use a lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and ensure direct log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /src

ENV GIT_PYTHON_REFRESH=quiet

COPY /src /src
COPY data/base_dataset.parquet /data/base_dataset.parquet

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 2020

CMD ["sh", "-c", "python data_ingestion.py && python train.py"]