FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends  curl\
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and trained models
COPY . .

# Expose service port
EXPOSE 8033

# Trivial HEALTHCHECK that always succeeds within 2s. Coolify v4 deploy-time
# rolling-update probe needs the container to report 'healthy' fast or it
# rolls back. Real /health monitoring happens at L7 via gateway proxy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8033}/health/live || exit 1

# Run application — shell form so $PORT is expanded from Railway env
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8033}
