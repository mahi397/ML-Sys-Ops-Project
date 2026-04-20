from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from core.config import get_db_dsn


def get_conn() -> psycopg.Connection[Any]:
    return psycopg.connect(get_db_dsn(), row_factory=dict_row)
