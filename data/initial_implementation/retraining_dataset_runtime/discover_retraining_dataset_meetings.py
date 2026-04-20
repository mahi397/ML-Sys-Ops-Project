#!/usr/bin/env python3
from __future__ import annotations

from feedback_common import get_conn


def main() -> None:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT m.meeting_id
            FROM meetings m
            JOIN feedback_events fe ON fe.meeting_id = m.meeting_id
            WHERE m.source_type = 'jitsi'
              AND m.is_valid = TRUE
              AND m.dataset_version IS NULL
              AND fe.event_type IN ('merge_segments', 'split_segment', 'boundary_correction')
            ORDER BY m.meeting_id
            """
        )
        rows = cur.fetchall()

    for row in rows:
        print(row["meeting_id"])


if __name__ == "__main__":
    main()
