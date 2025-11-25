# backend/Dockerfile (fixed)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install build deps needed to compile some Python packages (but avoid obsolete atlas package)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy python deps first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code & models
COPY . /app

# Ensure model files readable
RUN chmod -R a+r /app/models || true

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
