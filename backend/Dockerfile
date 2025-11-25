# backend/Dockerfile
FROM python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system deps required by some Python packages (numpy/pandas/scikit-learn/xgboost)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential gcc gfortran libatlas-base-dev libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy python deps first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# copy the application code (including models directory)
COPY . /app

# ensure models folder permissions are readable
RUN chmod -R a+r /app/models || true

EXPOSE 8000

# Use 0.0.0.0 so Render/other hosts can bind
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
