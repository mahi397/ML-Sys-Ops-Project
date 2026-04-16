FROM python:3.12-slim

WORKDIR /app

COPY jitsi_transcript_receiver.py /app/jitsi_transcript_receiver.py
COPY ingest_saved_jitsi_transcript.py /app/
COPY feedback_common.py /app/

RUN pip install --no-cache-dir fastapi uvicorn python-multipart

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "jitsi_transcript_receiver:app", "--host", "0.0.0.0", "--port", "9000"]
