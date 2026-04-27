# Proj07 Runtime

This folder contains the production data-runtime Python package and Docker image context. It no longer owns a Compose file or local env template.

Use the repository root for runtime orchestration:

```bash
cd ../..
docker compose up -d
```

Use the root `.env` for all configuration, including data workers, serving, training, and Jitsi deployment.

`traffic-generator` is intentionally manual-only:

```bash
docker compose --profile emulated-traffic up -d traffic-generator
```

The archived `data/initial_implementation/` tree is independent and is not used by the active runtime.
