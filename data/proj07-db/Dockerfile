FROM python:3.12-slim

WORKDIR /app

COPY jitsi_transcript_receiver.py /app/jitsi_transcript_receiver.py

RUN pip install --no-cache-dir fastapi uvicorn python-multipart

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "jitsi_transcript_receiver:app", "--host", "0.0.0.0", "--port", "9000"]
