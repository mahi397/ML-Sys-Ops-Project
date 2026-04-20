#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from proj07_services.pipeline.ingest_ami_meeting import (
    count_fully_ingested_meetings,
    get_ami_meeting_ingest_state,
    ingest_meeting,
)


APP_NAME = "bootstrap_ami_corpus"


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", nargs="*", help="Specific AMI meeting ids to process.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rclone-remote", default=os.getenv("RCLONE_REMOTE", "rclone_s3"))
    parser.add_argument(
        "--bucket",
        default=(
            os.getenv("OBJECT_BUCKET", "").strip()
            or os.getenv("BUCKET", "").strip()
            or "objstore-proj07"
        ),
    )
    parser.add_argument("--prefix", default=os.getenv("AMI_OBJECT_PREFIX", "ami_public_manual_1.6.2"))
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/block/staging/current_job/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("/mnt/block/staging/current_job/processed"))
    parser.add_argument("--processed-object-prefix", default="processed/ami/v1")
    parser.add_argument("--artifact-version", type=int, default=1)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("/mnt/block/ingest_logs/ami_corpus_bootstrap.log"))
    return parser.parse_args()


def get_conn():
    import psycopg
    from psycopg.rows import dict_row

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("Missing required environment variable: DATABASE_URL")
    return psycopg.connect(database_url, row_factory=dict_row)


def run_command(
    cmd: list[str],
    *,
    logger: logging.Logger,
    label: str,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    logger.info("Running %s: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=capture, check=False)
    if result.returncode != 0:
        if capture and result.stdout:
            logger.error("stdout:\n%s", result.stdout)
        if capture and result.stderr:
            logger.error("stderr:\n%s", result.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return result


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remote_path(remote: str, bucket: str, prefix: str, relative: str) -> str:
    return f"{remote}:{bucket}/{prefix}/{relative}"


def list_remote_files(remote: str, bucket: str, prefix: str, relative_dir: str, logger: logging.Logger) -> list[str]:
    result = run_command(
        ["rclone", "lsf", "--files-only", remote_path(remote, bucket, prefix, relative_dir)],
        logger=logger,
        capture=True,
        label=f"list remote files in {relative_dir}",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def discover_meeting_ids(remote: str, bucket: str, prefix: str, logger: logging.Logger) -> list[str]:
    topic_files = list_remote_files(remote, bucket, prefix, "topics", logger)
    meetings = sorted(
        {
            file_name[: -len(".topic.xml")]
            for file_name in topic_files
            if file_name.endswith(".topic.xml")
        }
    )
    logger.info("Discovered %d AMI meeting(s) from object storage", len(meetings))
    return meetings


def ensure_shared_cache(raw_root: Path, remote: str, bucket: str, prefix: str, logger: logging.Logger) -> None:
    required_files = [
        ("corpusResources/meetings.xml", raw_root / "corpusResources" / "meetings.xml"),
        ("corpusResources/participants.xml", raw_root / "corpusResources" / "participants.xml"),
        ("ontologies/default-topics.xml", raw_root / "ontologies" / "default-topics.xml"),
    ]

    for relative_key, local_path in required_files:
        ensure_dir(local_path.parent)
        if local_path.exists():
            continue
        run_command(
            [
                "rclone",
                "copyto",
                remote_path(remote, bucket, prefix, relative_key),
                str(local_path),
                "-P",
            ],
            logger=logger,
            label=f"cache shared file {relative_key}",
        )


def clear_meeting_specific_staging(raw_root: Path, processed_root: Path) -> None:
    for subdir in ("words", "segments", "topics", "abstractive"):
        path = raw_root / subdir
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    if processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)


def cleanup_meeting_specific(raw_root: Path, processed_root: Path) -> None:
    for subdir in ("words", "segments", "topics", "abstractive"):
        path = raw_root / subdir
        if path.exists():
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    if processed_root.exists():
        shutil.rmtree(processed_root)


def copy_meeting_specific_files(
    remote: str,
    bucket: str,
    prefix: str,
    raw_root: Path,
    meeting_id: str,
    logger: logging.Logger,
) -> None:
    mapping = [
        ("words", f"{meeting_id}*.xml"),
        ("segments", f"{meeting_id}*.xml"),
        ("topics", f"{meeting_id}.topic.xml"),
        ("abstractive", f"{meeting_id}.abssumm.xml"),
    ]

    for subdir, include_pattern in mapping:
        run_command(
            [
                "rclone",
                "copy",
                remote_path(remote, bucket, prefix, subdir),
                str(raw_root / subdir),
                "--include",
                include_pattern,
                "-P",
            ],
            logger=logger,
            label=f"stage {subdir} for {meeting_id}",
        )


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_file)

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    ensure_dir(raw_root)
    ensure_dir(processed_root)

    ensure_shared_cache(raw_root, args.rclone_remote, args.bucket, args.prefix, logger)

    if args.meeting:
        meetings = sorted(set(args.meeting))
        logger.info("Using explicit AMI meeting list: %s", ", ".join(meetings))
    else:
        meetings = discover_meeting_ids(args.rclone_remote, args.bucket, args.prefix, logger)

    if args.limit is not None:
        meetings = meetings[: args.limit]
        logger.info("Applied limit=%d, remaining meetings=%d", args.limit, len(meetings))

    if not meetings:
        logger.info("No AMI meetings selected; nothing to do")
        return 0

    with get_conn() as conn:
        complete_before = count_fully_ingested_meetings(conn, meetings)
    logger.info(
        "AMI bootstrap coverage before run | selected=%d fully_ingested=%d missing=%d",
        len(meetings),
        complete_before,
        len(meetings) - complete_before,
    )

    if complete_before == len(meetings) and not args.no_skip_existing:
        logger.info("Selected AMI meetings are already fully ingested in Postgres")
        return 0

    processed = 0
    skipped = 0
    failed = 0

    for idx, meeting_id in enumerate(meetings, start=1):
        logger.info("=" * 80)
        logger.info("[%d/%d] Processing AMI meeting=%s", idx, len(meetings), meeting_id)

        try:
            with get_conn() as conn:
                meeting_exists, fully_ingested = get_ami_meeting_ingest_state(conn, meeting_id)

            if fully_ingested and not args.no_skip_existing:
                logger.info("Skipping %s because it is already fully ingested", meeting_id)
                skipped += 1
                continue

            current_processed = processed_root / meeting_id
            clear_meeting_specific_staging(raw_root, current_processed)
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
                status = ingest_meeting(
                    meeting_id=meeting_id,
                    raw_root=raw_root,
                    processed_root=current_processed,
                    raw_folder_prefix=args.prefix,
                    processed_object_prefix=args.processed_object_prefix,
                    artifact_version=args.artifact_version,
                    replace_existing=meeting_exists,
                    cleanup_local_artifacts=not args.keep_staging,
                    rclone_remote=args.rclone_remote,
                    bucket=args.bucket,
                    logger=logger,
                )
                logger.info("AMI meeting ingest finished | meeting_id=%s | status=%s", meeting_id, status)

            if args.keep_staging:
                logger.info("Keeping staged files for %s", meeting_id)
            else:
                cleanup_meeting_specific(raw_root, current_processed)

            processed += 1
        except Exception:
            failed += 1
            logger.exception("Failed while processing AMI meeting=%s", meeting_id)
            logger.info("Leaving current staging in place for debugging")

    logger.info("=" * 80)

    if not args.dry_run:
        with get_conn() as conn:
            complete_after = count_fully_ingested_meetings(conn, meetings)
        logger.info(
            "AMI bootstrap coverage after run | selected=%d fully_ingested=%d missing=%d",
            len(meetings),
            complete_after,
            len(meetings) - complete_after,
        )
        if complete_after < len(meetings):
            logger.error(
                "AMI bootstrap incomplete: only %d/%d selected meetings are fully ingested",
                complete_after,
                len(meetings),
            )
            return 1

    logger.info("AMI bootstrap complete | processed=%d skipped=%d failed=%d", processed, skipped, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
