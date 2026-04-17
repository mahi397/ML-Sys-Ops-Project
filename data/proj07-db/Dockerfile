FROM python:3.12-slim

WORKDIR /app

COPY jitsi_transcript_receiver.py /app/jitsi_transcript_receiver.py
COPY ingest_saved_jitsi_transcript.py /app/
COPY build_online_inference_payloads.py /app/
COPY db_task_worker.py /app/
COPY stage1_payload_service.py /app/
COPY stage1_forward_service.py /app/
COPY feedback_common.py /app/
COPY task_service_common.py /app/
COPY workflow_task_common.py /app/

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates rclone \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn python-multipart "psycopg[binary]==3.2.1"

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "jitsi_transcript_receiver:app", "--host", "0.0.0.0", "--port", "9000"]
