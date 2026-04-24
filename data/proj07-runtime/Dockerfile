FROM python:3.12-slim

WORKDIR /app

COPY proj07_services /app/proj07_services

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates rclone \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn python-multipart requests "psycopg[binary]==3.2.1"

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "proj07_services.api.jitsi_transcript_receiver:app", "--host", "0.0.0.0", "--port", "9000"]
