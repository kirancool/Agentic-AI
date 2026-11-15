"""Restartable, bounded-memory MySQL -> PostgreSQL table loader.

This is the data-copy layer, not a schema converter. Create and review the target
schema first. For a consistent migration, stop writes to the source (or use a
separately coordinated snapshot/CDC plan) for the entire run.

Requires: mysql-connector-python, psycopg2-binary, python-dotenv
"""
from __future__ import annotations

import base64
import datetime as dt
import decimal
import json
import os
import time
from typing import Any, Iterable

import mysql.connector
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


MYSQL_CFG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.environ["MYSQL_USER"],
    "password": os.environ["MYSQL_PASSWORD"],
    "database": os.environ["MYSQL_DATABASE"],
    "connection_timeout": int(os.getenv("MYSQL_CONNECT_TIMEOUT", "30")),
}
PG_CFG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", "5432")),
    "user": os.environ["PG_USER"],
    "password": os.environ["PG_PASSWORD"],
    "dbname": os.environ["PG_DATABASE"],
    "connect_timeout": int(os.getenv("PG_CONNECT_TIMEOUT", "30")),
}

BATCH_SIZE = int(os.getenv("MIGRATION_BATCH_SIZE", "5000"))
CHECKPOINT_TABLE = "_migration_checkpoint"


def _mysql_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _checkpoint_encode(value: Any) -> Any:
    """Preserve PK types across a process restart."""
    if isinstance(value, bytes):
        return {"$type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, decimal.Decimal):
        return {"$type": "decimal", "value": str(value)}
    if isinstance(value, dt.datetime):
        return {"$type": "datetime", "value": value.isoformat()}
    if isinstance(value, dt.date):
        return {"$type": "date", "value": value.isoformat()}
    if isinstance(value, dt.time):
        return {"$type": "time", "value": value.isoformat()}
    return value


def _checkpoint_decode(value: Any) -> Any:
    if not isinstance(value, dict) or "$type" not in value:
        return value
    kind, raw = value["$type"], value["value"]
    if kind == "bytes": return base64.b64decode(raw)
    if kind == "decimal": return decimal.Decimal(raw)
    if kind == "datetime": return dt.datetime.fromisoformat(raw)
    if kind == "date": return dt.date.fromisoformat(raw)
    if kind == "time": return dt.time.fromisoformat(raw)
    raise ValueError(f"Unknown checkpoint type: {kind}")


def _get_primary_key(mysql_conn, table: str) -> list[str]:
    cur = mysql_conn.cursor()
    try:
        cur.execute(
            "SELECT COLUMN_NAME FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND INDEX_NAME='PRIMARY' "
            "ORDER BY SEQ_IN_INDEX",
            (MYSQL_CFG["database"], table),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        cur.close()


def _ensure_checkpoint_table(pg_conn) -> None:
    with pg_conn.cursor() as cur:
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                source_table text PRIMARY KEY,
                last_pk jsonb,
                rows_copied bigint NOT NULL DEFAULT 0,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
        """).format(sql.Identifier(CHECKPOINT_TABLE)))
    pg_conn.commit()


def _load_checkpoint(pg_conn, table: str) -> tuple[list[Any] | None, int]:
    with pg_conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT last_pk, rows_copied FROM {} WHERE source_table=%s")
                    .format(sql.Identifier(CHECKPOINT_TABLE)), (table,))
        row = cur.fetchone()
    if not row:
        return None, 0
    raw = row[0]
    if isinstance(raw, str):
        raw = json.loads(raw)
    return ([_checkpoint_decode(v) for v in raw] if raw is not None else None, row[1])


def _after_pk_clause(pk: list[str], last: list[Any]) -> tuple[str, list[Any]]:
    # Lexicographic keyset pagination works with composite primary keys.
    terms, params = [], []
    for i, col in enumerate(pk):
        equal_parts = [f"{_mysql_ident(pk[j])} = %s" for j in range(i)]
        params.extend(last[:i])
        terms.append("(" + " AND ".join(equal_parts + [f"{_mysql_ident(col)} > %s"]) + ")")
        params.append(last[i])
    return " OR ".join(terms), params


def migrate_table(table: str, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """Load/restart a table by ordered primary-key batches; each batch is atomic.

    Target conflict rows are updated, so a crash after a target commit but before
    client acknowledgement can safely replay that batch. This is an upsert-based
    initial load: it does not propagate source deletes or concurrent source edits.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    started = time.monotonic()
    mc = mysql.connector.connect(**MYSQL_CFG)
    pc = psycopg2.connect(**PG_CFG)
    source = None
    try:
        pk = _get_primary_key(mc, table)
        if not pk:
            raise ValueError(f"{table}: restartable loading requires a primary key")
        _ensure_checkpoint_table(pc)
        last_pk, total = _load_checkpoint(pc, table)

        # Server-side/unbuffered cursor: the whole source table is never fetched
        # into client RAM. Query batches by PK so each target commit is resumable.
        source = mc.cursor(buffered=False)
        source_columns: list[str] | None = None
        col_sql = "*"
        while True:
            where_sql, where_params = ("", [])
            if last_pk is not None:
                where_sql, where_params = _after_pk_clause(pk, last_pk)
                where_sql = " WHERE " + where_sql
            order_sql = ", ".join(_mysql_ident(c) for c in pk)
            source.execute(
                f"SELECT {col_sql} FROM {_mysql_ident(table)}{where_sql} "
                f"ORDER BY {order_sql} LIMIT %s",
                tuple(where_params + [batch_size]),
            )
            if source_columns is None:
                source_columns = [d[0] for d in source.description]
                lowered = [c.lower() for c in source_columns]
                if len(set(lowered)) != len(lowered):
                    raise ValueError(f"{table}: column names collide after PostgreSQL lowercasing")
                absent = set(c.lower() for c in pk) - set(lowered)
                if absent:
                    raise ValueError(f"{table}: primary key columns missing from projection: {absent}")
            rows = source.fetchall()
            if not rows:
                break

            cols = [c.lower() for c in source_columns]
            insert = sql.SQL("INSERT INTO {} ({}) VALUES %s ON CONFLICT ({}) DO UPDATE SET {}")
            target_table = sql.Identifier(table.lower())
            col_identifiers = sql.SQL(",").join(map(sql.Identifier, cols))
            pk_lower = [c.lower() for c in pk]
            pk_identifiers = sql.SQL(",").join(map(sql.Identifier, pk_lower))
            updates = [c for c in cols if c not in pk_lower]
            if updates:
                update_sql = sql.SQL(",").join(
                    sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
                    for c in updates
                )
                statement = insert.format(target_table, col_identifiers, pk_identifiers, update_sql)
            else:
                statement = sql.SQL("INSERT INTO {} ({}) VALUES %s ON CONFLICT ({}) DO NOTHING")
                statement = statement.format(target_table, col_identifiers, pk_identifiers)

            # MySQL zero dates are not representable as PostgreSQL dates.
            cleaned = []
            for row in rows:
                cleaned.append(tuple(None if isinstance(v, str) and v.startswith("0000-00-00") else v for v in row))
            positions = {name.lower(): i for i, name in enumerate(source_columns)}
            last_pk = [cleaned[-1][positions[c.lower()]] for c in pk]
            encoded_pk = json.dumps([_checkpoint_encode(v) for v in last_pk])
            with pc.cursor() as target:
                execute_values(target, statement.as_string(pc), cleaned, page_size=batch_size)
                target.execute(
                    sql.SQL("""INSERT INTO {} (source_table,last_pk,rows_copied,updated_at)
                               VALUES (%s,%s::jsonb,%s,now())
                               ON CONFLICT (source_table) DO UPDATE SET
                                 last_pk=EXCLUDED.last_pk,
                                 rows_copied={}.rows_copied + EXCLUDED.rows_copied,
                                 updated_at=now()""").format(sql.Identifier(CHECKPOINT_TABLE), sql.Identifier(CHECKPOINT_TABLE)),
                    (table, encoded_pk, len(cleaned)),
                )
            pc.commit()
            total += len(cleaned)
            print(f"{table}: committed {total:,} rows; last PK={last_pk!r}", flush=True)
            # Release the unbuffered result before issuing another query.
            source.fetchall()

        return {"table": table, "rows_this_run_and_prior_checkpoints": total,
                "last_pk": last_pk, "elapsed_seconds": round(time.monotonic() - started, 1)}
    except Exception:
        pc.rollback()
        raise
    finally:
        if source is not None:
            source.close()
        mc.close()
        pc.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Restartable MySQL to PostgreSQL bulk table loader")
    parser.add_argument("tables", nargs="+", help="MySQL base table names, in load order")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    for table in args.tables:
        print(json.dumps(migrate_table(table, args.batch_size), default=str))


if __name__ == "__main__":
    main()
