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

The installer downloads the upstream `docker-jitsi-meet` release into `/mnt/block/jitsi`, merges the root `.env` into the runtime Jitsi `.env`, generates missing secrets, and writes generated values such as `JWT_APP_SECRET`, `MEETING_PORTAL_SESSION_SECRET`, `INGEST_TOKEN`, and `JITSI_HOST_EXTERNAL_KEY` back to the root `.env` when possible.

The installer still uses upstream Jitsi compose files internally because that is how the Jitsi project is packaged. Project services remain controlled from the root `docker-compose.yml`; Jitsi deployment configuration remains controlled from the root `.env`.
