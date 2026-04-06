#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("run_ingest")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# -----------------------------
# Helpers
# -----------------------------
def run_cmd(
    cmd: List[str],
    logger: logging.Logger,
    capture: bool = False,
    check: bool = True,
    label: str | None = None,
) -> subprocess.CompletedProcess:
    label = label or "command"
    logger.info("Running %s: %s", label, " ".join(cmd))
    start = time.time()

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=capture,
        check=False,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error("Failed %s in %.2fs", label, elapsed)
        if capture:
            logger.error("stdout:\n%s", result.stdout)
            logger.error("stderr:\n%s", result.stderr)
        if check:
            raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    else:
        logger.info("Completed %s in %.2fs", label, elapsed)

    return result


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remote_path(remote: str, bucket: str, prefix: str, relative: str) -> str:
    return f"{remote}:{bucket}/{prefix}/{relative}"


def list_remote_files(remote: str, bucket: str, prefix: str, relative_dir: str, logger: logging.Logger) -> List[str]:
    rp = remote_path(remote, bucket, prefix, relative_dir)
    res = run_cmd(
        ["rclone", "lsf", "--files-only", rp],
        logger=logger,
        capture=True,
        label=f"list remote files in {relative_dir}",
    )
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def discover_meeting_ids(remote: str, bucket: str, prefix: str, logger: logging.Logger) -> List[str]:
    logger.info("Discovering meeting ids from topics/*.topic.xml")
    topic_files = list_remote_files(remote, bucket, prefix, "topics", logger)
    meetings = sorted({
        f[:-len(".topic.xml")]
        for f in topic_files
        if f.endswith(".topic.xml")
    })
    logger.info("Discovered %d meeting(s)", len(meetings))
    return meetings


def verify_shared_cache(raw_root: Path, logger: logging.Logger) -> None:
    logger.info("Verifying shared AMI cache under %s", raw_root)
    required = [
        raw_root / "corpusResources" / "meetings.xml",
        raw_root / "corpusResources" / "participants.xml",
        raw_root / "ontologies" / "default-topics.xml",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.error("Missing shared cached files:\n%s", "\n".join(missing))
        raise FileNotFoundError("Shared cached files are missing")
    logger.info("Shared AMI cache verified")


def clear_meeting_specific_staging(raw_root: Path, processed_root: Path, logger: logging.Logger) -> None:
    logger.info("Clearing local meeting-specific staging")
    for sub in ["words", "segments", "topics", "abstractive"]:
        p = raw_root / sub
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    if processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    logger.info("Meeting-specific staging directories are ready")


def copy_meeting_specific_files(
    remote: str,
    bucket: str,
    prefix: str,
    raw_root: Path,
    meeting_id: str,
    logger: logging.Logger,
) -> None:
    logger.info("Copying meeting-specific files for %s", meeting_id)

    mapping = [
        ("words", f"{meeting_id}*.xml"),
        ("segments", f"{meeting_id}*.xml"),
        ("topics", f"{meeting_id}.topic.xml"),
        ("abstractive", f"{meeting_id}.abssumm.xml"),
    ]

    for subdir, include_pattern in mapping:
        src = remote_path(remote, bucket, prefix, subdir)
        dst = raw_root / subdir
        run_cmd(
            [
                "rclone", "copy",
                src,
                str(dst),
                "--include", include_pattern,
                "-P",
            ],
            logger=logger,
            capture=False,
            label=f"copy {subdir} for {meeting_id}",
        )

    logger.info("Finished staging meeting-specific files for %s", meeting_id)


def meeting_exists_in_db(
    pg_container: str,
    db_user: str,
    db_name: str,
    meeting_id: str,
    logger: logging.Logger,
) -> bool:
    sql = f"SELECT 1 FROM meetings WHERE meeting_id = '{meeting_id}' LIMIT 1;"
    res = run_cmd(
        [
            "docker", "exec", pg_container,
            "psql", "-U", db_user, "-d", db_name,
            "-tAc", sql,
        ],
        logger=logger,
        capture=True,
        check=False,
        label=f"check existing meeting {meeting_id}",
    )
    exists = res.returncode == 0 and res.stdout.strip() == "1"
    logger.info("Meeting %s exists in Postgres: %s", meeting_id, exists)
    return exists


def cleanup_meeting_specific(raw_root: Path, processed_root: Path, logger: logging.Logger) -> None:
    logger.info("Cleaning meeting-specific local staging")
    for sub in ["words", "segments", "topics", "abstractive"]:
        p = raw_root / sub
        if p.exists():
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    if processed_root.exists():
        shutil.rmtree(processed_root)

    logger.info("Meeting-specific local staging cleaned")


def process_meeting(
    scripts_dir: Path,
    meeting_id: str,
    raw_root: Path,
    processed_root: Path,
    pg_container: str,
    db_user: str,
    db_name: str,
    rclone_remote: str,
    bucket: str,
    logger: logging.Logger,
) -> None:
    worker = scripts_dir / "ingest_one_meeting.py"
    if not worker.exists():
        raise FileNotFoundError(f"{worker} not found")

    log_file = Path("/mnt/block/ingest_logs") / f"{meeting_id}.log"
    ensure_dir(log_file.parent)

    run_cmd(
        [
            "python3", str(worker),
            "--meeting", meeting_id,
            "--raw-root", str(raw_root),
            "--processed-root", str(processed_root),
            "--pg-container", pg_container,
            "--db-user", db_user,
            "--db-name", db_name,
            "--rclone-remote", rclone_remote,
            "--bucket", bucket,
            "--log-file", str(log_file),
        ],
        logger=logger,
        capture=False,
        label=f"ingest meeting {meeting_id}",
    )


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", nargs="*", help="Specific meeting IDs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rclone-remote", default="rclone_s3")
    parser.add_argument("--bucket", default="objstore-proj07")
    parser.add_argument("--prefix", default="ami_public_manual_1.6.2")
    parser.add_argument("--raw-root", default="/mnt/block/staging/current_job/raw")
    parser.add_argument("--processed-root", default="/mnt/block/staging/current_job/processed")
    parser.add_argument("--scripts-dir", default="/mnt/block/scripts")
    parser.add_argument("--pg-container", default="postgres")
    parser.add_argument("--db-user", default="proj07_user")
    parser.add_argument("--db-name", default="proj07_sql_db")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("/mnt/block/ingest_logs/run_ingest.log"))
    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    scripts_dir = Path(args.scripts_dir)

    ensure_dir(raw_root)
    ensure_dir(processed_root)
    if not scripts_dir.exists():
        raise FileNotFoundError(f"scripts_dir does not exist: {scripts_dir}")

    logger.info("Starting run_ingest")
    logger.info("raw_root=%s", raw_root)
    logger.info("processed_root=%s", processed_root)
    logger.info("scripts_dir=%s", scripts_dir)
    logger.info("object_storage=%s:%s/%s", args.rclone_remote, args.bucket, args.prefix)
    logger.info("postgres_container=%s db=%s user=%s", args.pg_container, args.db_name, args.db_user)

    verify_shared_cache(raw_root, logger)

    if args.meeting:
        meetings = sorted(set(args.meeting))
        logger.info("Using explicit meeting list: %s", meetings)
    else:
        meetings = discover_meeting_ids(args.rclone_remote, args.bucket, args.prefix, logger)

    if args.limit is not None:
        meetings = meetings[:args.limit]
        logger.info("Applied limit=%d, remaining meetings=%d", args.limit, len(meetings))

    if not meetings:
        logger.info("No meetings found. Exiting.")
        return 0

    processed = 0
    skipped = 0
    failed = 0

    for idx, meeting_id in enumerate(meetings, start=1):
        logger.info("=" * 80)
        logger.info("[%d/%d] Processing meeting=%s", idx, len(meetings), meeting_id)

        try:
            if not args.no_skip_existing and meeting_exists_in_db(
                args.pg_container,
                args.db_user,
                args.db_name,
                meeting_id,
                logger,
            ):
                logger.info("Skipping %s because it already exists in Postgres", meeting_id)
                skipped += 1
                continue

            current_processed = processed_root / meeting_id

            clear_meeting_specific_staging(raw_root, current_processed, logger)
            copy_meeting_specific_files(
                args.rclone_remote,
                args.bucket,
                args.prefix,
                raw_root,
                meeting_id,
                logger,
            )

            if args.dry_run:
                logger.info("Dry run only: staged files for %s", meeting_id)
            else:
                process_meeting(
                    scripts_dir=scripts_dir,
                    meeting_id=meeting_id,
                    raw_root=raw_root,
                    processed_root=current_processed,
                    pg_container=args.pg_container,
                    db_user=args.db_user,
                    db_name=args.db_name,
                    rclone_remote=args.rclone_remote,
                    bucket=args.bucket,
                    logger=logger,
                )

            if args.keep_staging:
                logger.info("Keeping local staging for %s", meeting_id)
            else:
                cleanup_meeting_specific(raw_root, current_processed, logger)

            processed += 1
            logger.info("Finished meeting=%s successfully", meeting_id)

        except Exception:
            failed += 1
            logger.exception("Failed while processing meeting=%s", meeting_id)
            logger.info("Leaving current staging in place for debugging")
            continue

    logger.info("=" * 80)
    logger.info("INGEST RUN COMPLETE")
    logger.info("Processed=%d", processed)
    logger.info("Skipped=%d", skipped)
    logger.info("Failed=%d", failed)
    logger.info("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
