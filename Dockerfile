# 1. Use a stable Python version
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (needed for ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first (for layer caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the app
COPY . .

# 7. Create non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# 8. Expose port
EXPOSE 8000

# 9. Start the app
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8000}"]
