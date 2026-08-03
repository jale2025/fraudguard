FROM python:3.11.13-slim-bookworm

# Prevent Python from writing .pyc files and ensure direct log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /webservice

COPY requirements.txt /webservice/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /webservice

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
