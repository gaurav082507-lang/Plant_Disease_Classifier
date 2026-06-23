FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Install foundational runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency layer structures cleanly
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Sync application domains and artifact data layers
COPY ./app /code/app
COPY ./models /code/models

EXPOSE 8000

# Exec form execution ensures clean termination signal mapping (SIGTERM)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
