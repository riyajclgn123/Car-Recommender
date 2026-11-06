FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so Docker caches this layer
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# App code (src/ includes the TCO lookup tables)
COPY api/ api/
COPY src/ src/

# Only the model files the API uses (not the PyTorch MLP)
COPY models/lgbm_tuned.pkl models/lifestage_classifier.pkl \
     models/lifestage_label_encoder.pkl models/lifestage_meta.json models/

# Feature metadata + slim inventory
COPY data/processed/feature_meta.json data/processed/inventory_deploy.parquet data/processed/

ENV INVENTORY_PATH=data/processed/inventory_deploy.parquet

EXPOSE 8080
# Render sets $PORT; fall back to 8080 locally
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}