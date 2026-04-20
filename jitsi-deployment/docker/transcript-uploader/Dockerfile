FROM python:3.12-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY assets/scripts/send_completed_jitsi_transcripts.py /app/send_completed_jitsi_transcripts.py

CMD ["python", "-u", "/app/send_completed_jitsi_transcripts.py"]
