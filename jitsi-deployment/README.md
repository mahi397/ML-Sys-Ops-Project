# Jitsi Deployment

Jitsi uses the repository root `.env` as its source of configuration. There is no separate `stack.env`.

From the repo root, deploy Jitsi with:

```bash
DEPLOY_JITSI=true bash setup.sh
```

Or run the installer directly:

```bash
sudo bash jitsi-deployment/install-jitsi-vm.sh --env-file .env
```

The installer downloads the upstream `docker-jitsi-meet` release into
`/mnt/block/jitsi`, merges the root `.env` into the runtime Jitsi `.env`,
derives the meeting-portal DB DSN and transcript ingest URL from the shared
Postgres/data host settings when needed, generates missing secrets, and writes
generated values such as `JWT_APP_SECRET`, `MEETING_PORTAL_SESSION_SECRET`,
`INGEST_TOKEN`, and `JITSI_HOST_EXTERNAL_KEY` back to the root `.env` when
possible.

The installer still uses upstream Jitsi compose files internally because that is how the Jitsi project is packaged. Project services remain controlled from the root `docker-compose.yml`; Jitsi deployment configuration remains controlled from the root `.env`.

The installer runs Jitsi as the Docker Compose project `jitsi-vm`. When checking
the installed stack manually from `/mnt/block/jitsi/jitsi-docker-jitsi-meet`,
include the same project name or Compose will show an empty project:

```bash
docker compose --project-name jitsi-vm \
  -f docker-compose.yml \
  -f jigasi.yml \
  -f transcriber.yml \
  -f jitsi-deployment/compose/vm-services.yml \
  ps

docker compose --project-name jitsi-vm \
  -f docker-compose.yml \
  -f jigasi.yml \
  -f transcriber.yml \
  -f jitsi-deployment/compose/vm-services.yml \
  logs --since 15m prosody jicofo jigasi transcriber vosk transcript-uploader
```

## Manual Service Operations

Run these from `/mnt/block/jitsi/jitsi-docker-jitsi-meet` after the installer has
finished:

| Service | Manual run | Logs | Stop |
|---------|------------|------|------|
| `web` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d web` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m web` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop web` |
| `prosody` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d prosody` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m prosody` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop prosody` |
| `jicofo` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d jicofo` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m jicofo` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop jicofo` |
| `jvb` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d jvb` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m jvb` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop jvb` |
| `jigasi` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d jigasi` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m jigasi` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop jigasi` |
| `transcriber` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d transcriber` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m transcriber` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop transcriber` |
| `meeting-portal-app` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d meeting-portal-app` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m meeting-portal-app` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop meeting-portal-app` |
| `transcript-uploader` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d transcript-uploader` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m transcript-uploader` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop transcript-uploader` |
| `vosk` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml up -d vosk` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml logs -f --since 15m vosk` | `docker compose --project-name jitsi-vm -f docker-compose.yml -f jigasi.yml -f transcriber.yml -f jitsi-deployment/compose/vm-services.yml stop vosk` |
