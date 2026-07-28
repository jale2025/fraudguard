FROM python:3.11-slim

# Prevent Python from writing .pyc files and ensure direct log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/prefect

WORKDIR /prefect

COPY prefect/requirements.txt /prefect/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /prefect/src
COPY prefect /prefect
COPY dbt /prefect/dbt

CMD ["python", "flow.py"]