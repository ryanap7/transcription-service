# ============================================================================
# SIMTELPAS AI
# ============================================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create python symlinks
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN python -m pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install other packages
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/outputs /app/temp && \
    chown -R appuser:appuser /app/outputs /app/temp

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV FLASK_ENV=production

# Expose port
EXPOSE 8000

# Healthcheck - increased start_period for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run with Gunicorn using SYNC workers
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--worker-class", "sync", \
     "--timeout", "900", \
     "--max-requests", "100", \
     "--max-requests-jitter", "10", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--preload", \
     "wsgi:app"]