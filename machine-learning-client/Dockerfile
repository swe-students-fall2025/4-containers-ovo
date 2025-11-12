FROM python:3.11-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install audio reading dependencies
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY app ./app

# Start command: Run worker.py
CMD ["python", "-m", "app.worker"]
