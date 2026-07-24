FROM python:3.11-slim

# Prevent Python from writing .pyc files and ensure direct log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/prefect/src

WORKDIR /prefect

COPY src /prefect/src
COPY prefect /prefect
COPY dbt /prefect/dbt

RUN pip install --no-cache-dir -r /prefect/requirements.txt

CMD ["python", "flow.py"]