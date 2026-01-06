# ============================================================================
# DOCKERFILE - STOCK ANALYSIS UI
# Build: docker build -t stock-analysis:latest .
# Run:   docker run -p 5000:5000 --env-file .env stock-analysis:latest
# ============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="Stock Analysis Team"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env.example .env

# Create necessary directories
RUN mkdir -p cache_data data_cache cache_logs signaux logs

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command (API server with gunicorn)
# Pour développement: python src/api.py
# Pour production: gunicorn
# Utiliser PORT de l'environnement (Render utilise $PORT dynamique)
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 4 --threads 2 --worker-class gthread --timeout 120 --access-logfile - --error-logfile - src.api:app

# Alternative: démarrer en mode développement
# CMD ["python", "src/api.py"]
