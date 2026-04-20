# Standalone Jitsi VM Deployment

This folder is a standalone deployment bundle.

You can place only `jitsi-deployment/` inside your own GitHub repo, clone that repo on the VM, and run the installer from this folder. The installer downloads the latest upstream `jitsi/docker-jitsi-meet` release, prepares `.env`, generates passwords, adds the custom `meeting-portal-app` and `transcript-uploader` services, and starts the full stack.

## VM layout

- install root: `/mnt/block/jitsi`
- upstream Jitsi source: `/mnt/block/jitsi/jitsi-docker-jitsi-meet`
- Jitsi config root: `/mnt/block/jitsi/config`

## What the installer does

The script follows the same flow you described, but automates it:

1. fetches the latest upstream `docker-jitsi-meet` release zip from GitHub using the same idea as:

```bash
wget $(wget -q -O - https://api.github.com/repos/jitsi/docker-jitsi-meet/releases/latest | grep zip | cut -d\" -f4)
```

2. unzips it under `/mnt/block/jitsi`,
3. copies this `jitsi-deployment/` bundle into the downloaded source tree,
4. runs the equivalent of:

```bash
cp env.example .env
./gen-passwords.sh
```

5. creates the Jitsi config/data directories under `/mnt/block/jitsi/config`,
6. merges your deployment values from `stack.env`,
7. generates missing secrets such as:
- `MEETING_PORTAL_SESSION_SECRET`
- `JWT_APP_SECRET`
- `INGEST_TOKEN`
- `JITSI_HOST_EXTERNAL_KEY`
8. validates the generated `.env` before startup, including:
- auth and prosody wiring
- Jigasi SIP settings
- transcription and Vosk settings
- transcript uploader settings
- meeting-portal service settings
9. downloads the Vosk model if it is missing,
10. builds the custom images for:
- `web`
- `prosody`
- `meeting-portal-app`
- `transcript-uploader`
11. starts Jitsi with:

```bash
docker compose \
  -f docker-compose.yml \
  -f jigasi.yml \
  -f transcriber.yml \
  -f jitsi-deployment/compose/vm-services.yml \
  up -d --build
```

## Why this avoids the executable-bit issue

The earlier breakage came from manually copied files losing execute permissions on the VM.

This deployment avoids that by:

- downloading a fresh upstream release each time,
- normalizing script permissions after extraction,
- baking the custom Prosody/Web hooks into images instead of depending on VM bind-mounted executable init scripts.

## Files

- `install-jitsi-vm.sh`
  One-command VM installer.
- `stack.env.example`
  Deployment environment template.
- `compose/vm-services.yml`
  Compose overlay for the custom services.
- `assets/`
  Self-contained copy of the meeting portal app and transcript uploader assets.

## First-time setup

1. Copy the env template:

```bash
cp jitsi-deployment/stack.env.example jitsi-deployment/stack.env
```

2. Edit `jitsi-deployment/stack.env`.

Required values:

- `MEETING_PORTAL_DATABASE_URL`
- `JITSI_TRANSCRIPT_INGEST_URL`

Recommended values:

- `PUBLIC_URL`
- `JVB_ADVERTISE_IPS`
- `JITSI_HOST_EXTERNAL_KEY`

Also replace the demo `JIGASI_SIP_*` values with real SIP provider details if you want Jigasi to start successfully. The installer now validates those fields and will stop if they are still left at the sample `sip2sip.info` defaults.

If `PUBLIC_URL` or `JVB_ADVERTISE_IPS` are left blank, the installer will try to infer them from the VM IP.

3. If the meeting portal needs recap artifact access through `rclone`, place:

```text
/mnt/block/jitsi/config/rclone/rclone.conf
```

The app can still start without it, but artifact lookups that depend on `rclone` will fail until it exists.

## One command

Run this from your cloned repo root:

```bash
sudo bash jitsi-deployment/install-jitsi-vm.sh --env-file jitsi-deployment/stack.env
```

Or from inside this folder:

```bash
sudo bash install-jitsi-vm.sh --env-file stack.env
```

## Services started

- base Jitsi stack from upstream `docker-compose.yml`
- SIP gateway from upstream `jigasi.yml`
- transcription sidecar from upstream `transcriber.yml`
- `vosk`
- `meeting-portal-app`
- `transcript-uploader`

## Notes

- The installer assumes a Linux VM with persistent storage mounted at `/mnt/block`.
- Automatic Docker installation is implemented for Ubuntu/Debian only.
- If `ENABLE_LETSENCRYPT=1`, set valid `PUBLIC_URL`, `LETSENCRYPT_DOMAIN`, and `LETSENCRYPT_EMAIL` before running the installer.
- If you leave `JITSI_HOST_EXTERNAL_KEY` blank, the installer generates one with `uuidgen` and writes it into the deployed `.env`.
