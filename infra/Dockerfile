FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY router_service/ ./router_service/
COPY evaluator_model/ ./evaluator_model/
COPY api_server/ ./api_server/
COPY data_pipeline/ ./data_pipeline/
COPY evaluation/ ./evaluation/

# Create checkpoint directory
RUN mkdir -p checkpoints

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the API server
CMD ["python", "-m", "api_server.main"]
