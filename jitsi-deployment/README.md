# Standalone Jitsi VM Deployment

This folder is a standalone deployment bundle.

You can place only `jitsi-deployment/` inside your own GitHub repo, clone that repo on the VM, and run the installer from this folder. The installer downloads the latest upstream `jitsi/docker-jitsi-meet` release, prepares `.env`, generates passwords, adds the custom `meeting-portal-app` and `transcript-uploader` services, and starts the full stack.

## VM layout

- install root: `/mnt/block/jitsi`
- upstream Jitsi source: `/mnt/block/jitsi/jitsi-docker-jitsi-meet`
- Jitsi config root: `/mnt/block/jitsi/config`

## What the installer does

The script follows the same flow you described, but automates it:

1. fetches the latest upstream `docker-jitsi-meet` release zip from GitHub using the release API's `zipball_url`,

```bash
wget "$(wget -q -O - https://api.github.com/repos/jitsi/docker-jitsi-meet/releases/latest | sed -n 's/.*"zipball_url": "\(.*\)".*/\1/p' | head -n 1)"
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

Update these values before running the installer:

- `PUBLIC_URL`
- `JVB_ADVERTISE_IPS`
- `MEETING_PORTAL_DATABASE_URL`
- `JITSI_TRANSCRIPT_INGEST_URL`

Example values if your Jitsi VM is `198.51.100.25` and your runtime/API VM is `198.51.100.30`:

```dotenv
PUBLIC_URL=https://198.51.100.25:8443
JVB_ADVERTISE_IPS=198.51.100.25
MEETING_PORTAL_DATABASE_URL=postgresql://proj07_user:proj07@198.51.100.30:5432/proj07_sql_db
JITSI_TRANSCRIPT_INGEST_URL=http://198.51.100.30:9000/ingest/jitsi-transcript
```

Recommended values:

- `JITSI_HOST_EXTERNAL_KEY`

The bundled `stack.env.example` now defaults to `HTTP_PORT=8000` and `HTTPS_PORT=8443`. If you keep
those ports, make sure `PUBLIC_URL` includes the external HTTPS port, for example
`https://YOUR_VM_IP:8443`. If you change ports, keep `PUBLIC_URL`, `HTTP_PORT`, `HTTPS_PORT`, and
`JVB_ADVERTISE_IPS` aligned with how the VM is actually reachable.

For transcription-only installs, leave `JIGASI_DISABLE_SIP=1`. Only set `JIGASI_DISABLE_SIP=0` and replace the demo `JIGASI_SIP_*` values if you want the SIP/PSTN gateway itself.

If `PUBLIC_URL` or `JVB_ADVERTISE_IPS` are left blank, the installer will try to infer them from the VM IP.

3. If the meeting portal needs recap artifact access through `rclone`, place:

```text
/mnt/block/jitsi/config/rclone/rclone.conf
```

If you run the installer with `sudo` and already have `~/.config/rclone/rclone.conf` for that user on the VM, the installer will copy it into the deployed config root automatically. The app can still start without it, but artifact lookups that depend on `rclone` will fail until it exists.

## One command

Run this from your cloned repo root:

```bash
sudo bash jitsi-deployment/install-jitsi-vm.sh --env-file jitsi-deployment/stack.env
```

Or from inside this folder:

```bash
sudo bash install-jitsi-vm.sh --env-file stack.env
```

After the installer finishes, use the same overlay set for any follow-up `docker compose`
command on the VM:

```bash
docker compose \
  --project-name jitsi-vm \
  -f docker-compose.yml \
  -f jigasi.yml \
  -f transcriber.yml \
  -f jitsi-deployment/compose/vm-services.yml \
  ps
```

Do not replace `jitsi-deployment/compose/vm-services.yml` with repo-root overlays such as
`meeting-portal-app.yml`, `transcript-uploader.yml`, or `vosk.yml` on the VM install created by
this bundle. The VM deployment folds those services into `vm-services.yml`.

## Services started

- base Jitsi stack from upstream `docker-compose.yml`
- SIP gateway from upstream `jigasi.yml`
- transcription sidecar from upstream `transcriber.yml`
- `vosk`
- `meeting-portal-app`
- `transcript-uploader`

## Troubleshooting

If `docker logs` reports `No such container` for a transcriber or vosk container, first confirm the
Compose project name and active services:

```bash
docker compose \
  --project-name jitsi-vm \
  -f docker-compose.yml \
  -f jigasi.yml \
  -f transcriber.yml \
  -f jitsi-deployment/compose/vm-services.yml \
  ps
```

By default the installer uses project name `jitsi-vm`, so the container names are typically
`jitsi-vm-transcriber-1` and `jitsi-vm-vosk-1`.

If meetings fail to start and the `vosk` service is restarting, inspect the Vosk and transcriber
logs:

```bash
docker logs --tail=200 jitsi-vm-vosk-1
docker logs --tail=200 jitsi-vm-transcriber-1
```

If the Vosk log says `/opt/vosk-model-en/model` does not contain model files, the Vosk model was
not downloaded or mounted correctly. Rerun the installer so it can repopulate `VOSK_MODEL_PATH` and
start the stack again:

```bash
sudo bash jitsi-deployment/install-jitsi-vm.sh --env-file jitsi-deployment/stack.env
```

If the installer stops immediately after printing `Fetching latest docker-jitsi-meet release metadata`,
that step failed before the archive download started. With the current installer, the most common causes
are outbound HTTPS access to `api.github.com`/`github.com` being blocked from the VM or an older copy of
the installer that still looked for the wrong GitHub API field.

## Notes

- The installer assumes a Linux VM with persistent storage mounted at `/mnt/block`.
- Automatic Docker installation is implemented for Ubuntu/Debian only.
- If `ENABLE_LETSENCRYPT=1`, set valid `PUBLIC_URL`, `LETSENCRYPT_DOMAIN`, and `LETSENCRYPT_EMAIL` before running the installer.
- If you leave `JITSI_HOST_EXTERNAL_KEY` blank, the installer generates one with `uuidgen` and writes it into the deployed `.env`.
