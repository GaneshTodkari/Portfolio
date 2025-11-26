FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Minimal system deps to build numpy/scipy/xgboost where required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc gfortran pkg-config ca-certificates \
    libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure models readable
RUN chmod -R a+r /app/models || true

EXPOSE 8000

# Run a single worker, minimal logging, JSON-style CMD
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "warning"]
