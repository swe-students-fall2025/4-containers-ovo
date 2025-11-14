FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg sox libsndfile1 && rm -rf /var/lib/apt/lists/*

# Non-root
RUN useradd -m appuser
USER appuser

WORKDIR /app

COPY --chown=appuser:appuser requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=appuser:appuser web-app/ web-app/

WORKDIR /app/web-app
ENV FLASK_APP=app.py PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK CMD python -c "import socket; s=socket.socket(); s.connect(('127.0.0.1',8000)); s.close()"

CMD ["bash","-lc","flask run --host 0.0.0.0 --port 8000"]