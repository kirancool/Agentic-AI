"""Python-only Aurora MySQL -> GCS -> Dataflow -> Cloud Spanner pipeline."""
from __future__ import annotations

import base64
import gzip
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pymysql
from google.cloud import spanner, storage


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ValueError(f"Required setting is missing: {name}")
    return value


def source_connection():
    kwargs = {
        "host": env("AURORA_HOST"), "port": int(os.getenv("AURORA_PORT", "3306")),
        "user": env("AURORA_USER"), "password": env("AURORA_PASSWORD"),
        "database": env("AURORA_DATABASE"), "charset": "utf8mb4", "autocommit": True,
        "connect_timeout": 30, "read_timeout": 3600, "write_timeout": 3600,
    }
    if os.getenv("AURORA_SSL", "true").lower() == "true":
        kwargs["ssl"] = {"ca": env("AURORA_SSL_CA")}
    return pymysql.connect(**kwargs)


def gcs_parts():
    return storage.Client(project=env("GOOGLE_CLOUD_PROJECT")), env("GCS_BUCKET"), env("GCS_PREFIX").strip("/")


def spanner_database():
    client = spanner.Client(project=env("GOOGLE_CLOUD_PROJECT"))
    return client.instance(env("SPANNER_INSTANCE_ID")).database(env("SPANNER_DATABASE_ID"))


def source_schema():
    connection = source_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE,
                                     NUMERIC_PRECISION, NUMERIC_SCALE, ORDINAL_POSITION
                              FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s
                              ORDER BY TABLE_NAME, ORDINAL_POSITION""", (env("AURORA_DATABASE"),))
            columns = cursor.fetchall()
            cursor.execute("""SELECT TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION
                              FROM information_schema.KEY_COLUMN_USAGE
                              WHERE TABLE_SCHEMA=%s AND CONSTRAINT_NAME='PRIMARY'
                              ORDER BY TABLE_NAME, ORDINAL_POSITION""", (env("AURORA_DATABASE"),))
            keys = cursor.fetchall()
        schema = {}
        for table, column, data_type, column_type, nullable, precision, scale, ordinal in columns:
            schema.setdefault(table, {"table": table, "columns": [], "primary_key": []})["columns"].append({
                "name": column, "data_type": data_type.lower(), "column_type": column_type.lower(),
                "nullable": nullable == "YES", "precision": precision, "scale": scale, "ordinal": ordinal,
            })
        for table, column, ordinal in keys:
            schema.setdefault(table, {"table": table, "columns": [], "primary_key": []})["primary_key"].append(column)
        for table, metadata in schema.items():
            if not metadata["primary_key"]:
                raise ValueError(f"Spanner requires a primary key; source table {table!r} has none")
            validate_identifier(table)
            for col in metadata["columns"]:
                validate_identifier(col["name"])
        return list(schema.values())
    finally:
        connection.close()


def validate_identifier(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Table/column identifier {name!r} needs an explicit supported-name mapping")


def spanner_type(column: dict) -> str:
    kind, ct = column["data_type"], column["column_type"]
    if kind == "tinyint" and re.search(r"tinyint\s*\(\s*1\s*\)", ct):
        return "BOOL"
    if kind in {"tinyint", "smallint", "mediumint", "int", "integer", "bigint", "year"}:
        if kind == "bigint" and "unsigned" in ct:
            return "STRING(MAX)"
        return "INT64"
    if kind in {"decimal", "numeric"}:
        return "NUMERIC" if (column.get("precision") or 0) <= 38 and (column.get("scale") or 0) <= 9 else "STRING(MAX)"
    if kind in {"float", "double", "real"}:
        return "FLOAT64"
    if kind == "date":
        return "DATE"
    if kind in {"datetime", "timestamp"}:
        return "TIMESTAMP"
    if kind == "json":
        return "JSON"
    if kind in {"binary", "varbinary", "tinyblob", "blob", "mediumblob", "longblob"}:
        return "BYTES(MAX)"
    if kind == "bit":
        return "BOOL" if re.search(r"bit\s*\(\s*1\s*\)", ct) else "BYTES(MAX)"
    if kind in {"char", "varchar", "tinytext", "text", "mediumtext", "longtext", "enum", "set", "time"}:
        return "STRING(MAX)"
    raise ValueError(f"Unsupported Aurora MySQL type {ct!r} for column {column['name']!r}")


def ddl_for_table(table: dict) -> str:
    definitions = []
    for column in table["columns"]:
        required = " NOT NULL" if not column["nullable"] or column["name"] in table["primary_key"] else ""
        definitions.append(f"`{column['name']}` {spanner_type(column)}{required}")
    keys = ", ".join(f"`{key}`" for key in table["primary_key"])
    return f"CREATE TABLE `{table['table']}` ({', '.join(definitions)}) PRIMARY KEY ({keys})"


def preflight() -> None:
    connection = source_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
    finally:
        connection.close()
    schema = source_schema()
    if not schema:
        raise RuntimeError("No base tables were found in Aurora MySQL")
    database = spanner_database()
    list(database.execute_sql("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"))
    client, bucket_name, _ = gcs_parts()
    if client.lookup_bucket(bucket_name) is None:
        raise RuntimeError(f"GCS bucket does not exist or is inaccessible: {bucket_name}")
    print(json.dumps({"preflight": "ok", "tables": len(schema), "spanner": env("SPANNER_INSTANCE_ID") + "/" + env("SPANNER_DATABASE_ID")}))


def ai_review() -> None:
    from openai import OpenAI
    schema = source_schema()
    review_input = [{"table": t["table"], "primary_key": t["primary_key"],
                     "columns": [{"name": c["name"], "mysql_type": c["column_type"], "nullable": c["nullable"]} for c in t["columns"]]}
                    for t in schema]
    response = OpenAI(api_key=env("OPENAI_API_KEY")).responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-astra"), reasoning={"effort": "high"},
        instructions=("You are the Aurora MySQL to Cloud Spanner schema-review agent. Review only table/column names, "
                      "types and primary keys. Do not expect or request row values. Identify type mapping, key, index/query, "
                      "schema compatibility, large-table throughput, and cutover risks. Return concise practical advice."),
        input="Review this metadata-only schema inventory; do not claim anything has been migrated:\n" + json.dumps(review_input),
    )
    report = {"model": os.getenv("OPENAI_MODEL", "gpt-6-astra"), "review": response.output_text,
              "scope": "table/column metadata only; no row values"}
    gcs, bucket, prefix = gcs_parts()
    gcs.bucket(bucket).blob(f"{prefix}/ai-review.json").upload_from_string(json.dumps(report, indent=2), content_type="application/json")
    print("GPT-6 Astra schema review saved to GCS.")


def ensure_schema() -> None:
    schema = source_schema()
    database = spanner_database()
    existing = {row[0] for row in database.execute_sql("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")}
    ddl = [ddl_for_table(t) for t in schema if t["table"] not in existing]
    gcs, bucket, prefix = gcs_parts()
    gcs.bucket(bucket).blob(f"{prefix}/spanner-ddl.sql").upload_from_string(";\n".join(ddl) + (";\n" if ddl else ""), content_type="text/plain")
    if ddl and os.getenv("SPANNER_AUTO_CREATE_SCHEMA", "true").lower() == "true":
        database.update_ddl(ddl).result(timeout=3600)
    elif ddl:
        raise RuntimeError(f"Spanner tables missing ({len(ddl)}). Apply gs://{bucket}/{prefix}/spanner-ddl.sql or set SPANNER_AUTO_CREATE_SCHEMA=true")
    # Check source columns exist in every target table before any row export.
    all_target_columns = {}
    for row in database.execute_sql("SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS"):
        all_target_columns.setdefault(row[0], set()).add(row[1])
    for table in schema:
        target_cols = all_target_columns.get(table["table"], set())
        absent = {c["name"] for c in table["columns"]} - target_cols
        if absent:
            raise RuntimeError(f"Target Spanner table {table['table']} lacks source columns: {sorted(absent)}")
    print(json.dumps({"schema": "ready", "tables_created": len(ddl)}))


def json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return {"$kind": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"$kind": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"$kind": "date", "value": value.isoformat()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$kind": "bytes", "value": base64.b64encode(bytes(value)).decode("ascii")}
    return str(value)


def export_to_gcs() -> None:
    schema = source_schema()
    gcs, bucket_name, prefix = gcs_parts()
    bucket = gcs.bucket(bucket_name)
    chunk_limit = int(os.getenv("EXPORT_CHUNK_MB", "128")) * 1024 * 1024
    batch_size = int(os.getenv("EXPORT_BATCH_SIZE", "1000"))
    gcs.bucket(bucket_name).blob(f"{prefix}/schema.json").upload_from_string(json.dumps(schema), content_type="application/json")
    counts = {}
    connection = source_connection()
    try:
        for table in schema:
            name = table["table"]
            columns = [c["name"] for c in table["columns"]]
            quoted = ", ".join("`" + c.replace("`", "``") + "`" for c in columns)
            cursor = connection.cursor(pymysql.cursors.SSCursor)
            cursor.execute(f"SELECT {quoted} FROM `{name}`")
            count = part = 0
            temp_path = stream = None

            def open_chunk():
                fd, path = tempfile.mkstemp(prefix="aurora-export-", suffix=".jsonl.gz")
                os.close(fd)
                return Path(path), gzip.open(path, "wt", encoding="utf-8")

            def upload(path, number):
                object_name = f"{prefix}/data/{name}/part-{number:08d}.jsonl.gz"
                bucket.blob(object_name).upload_from_filename(str(path), content_type="application/gzip")
                path.unlink(missing_ok=True)
                print(f"Uploaded gs://{bucket_name}/{object_name}", flush=True)

            try:
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    for row in rows:
                        if stream is None:
                            temp_path, stream = open_chunk()
                        record = {"table": name, "values": {col: json_value(val) for col, val in zip(columns, row)}}
                        stream.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                        stream.flush()
                        count += 1
                        if temp_path.stat().st_size >= chunk_limit:
                            stream.close(); upload(temp_path, part)
                            part += 1; temp_path, stream = None, None
                if stream is not None:
                    stream.close(); upload(temp_path, part)
            finally:
                cursor.close()
                if stream is not None:
                    stream.close()
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            counts[name] = count
            print(f"Exported {count} rows from {name}", flush=True)
    finally:
        connection.close()
    bucket.blob(f"{prefix}/export-manifest.json").upload_from_string(json.dumps(counts, indent=2), content_type="application/json")
    print(json.dumps({"exported_rows": sum(counts.values()), "tables": len(counts)}))


def load_to_spanner() -> None:
    import apache_beam as beam
    from apache_beam import pvalue
    from apache_beam.io.filesystem import CompressionTypes
    from apache_beam.metrics.metric import Metrics
    from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
    from apache_beam.transforms.util import BatchElements
    import time

    gcs, bucket, prefix = gcs_parts()
    schema = json.loads(gcs.bucket(bucket).blob(f"{prefix}/schema.json").download_as_text())
    table_schemas = {t["table"]: t for t in schema}

    class SpannerWriter(beam.DoFn):
        def __init__(self, project, instance_id, database_id, schemas):
            self.project, self.instance_id, self.database_id, self.schemas = project, instance_id, database_id, schemas
            self.ok = Metrics.counter("aurora_spanner", "rows_written")
            self.failed = Metrics.counter("aurora_spanner", "write_failures")

        def setup(self):
            client = spanner.Client(project=self.project)
            self.database = client.instance(self.instance_id).database(self.database_id)

        def decode(self, raw, column):
            if raw is None:
                return None
            kind = column["data_type"]
            if isinstance(raw, dict) and "$kind" in raw:
                value = raw["value"]
                tag = raw["$kind"]
                if tag == "decimal":
                    if spanner_type(column) == "NUMERIC": return Decimal(value)
                    return value
                if tag == "datetime":
                    parsed = datetime.fromisoformat(value)
                    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
                if tag == "date":
                    return date.fromisoformat(value)
                if tag == "bytes":
                    decoded = base64.b64decode(value)
                    return bool(int.from_bytes(decoded, "big")) if spanner_type(column) == "BOOL" else base64.b64encode(decoded)
            target = spanner_type(column)
            if target == "BOOL":
                if isinstance(raw, (bytes, bytearray)):
                    return bool(int.from_bytes(raw, "big"))
                return str(raw).lower() not in {"0", "false", ""}
            if target == "INT64": return int(raw)
            if target == "FLOAT64": return float(raw)
            if kind == "json":
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                return json.dumps(parsed, separators=(",", ":"))
            if kind in {"decimal", "numeric"} and target.startswith("STRING"): return str(raw)
            if target.startswith("STRING") and not isinstance(raw, str): return str(raw)
            return raw

        def process(self, records):
            grouped = {}
            for record in records:
                try:
                    row = json.loads(record) if isinstance(record, str) else record
                    table = row["table"]
                    metadata = self.schemas[table]
                    columns = [c["name"] for c in metadata["columns"]]
                    values = [self.decode(row["values"].get(col), colinfo) for col, colinfo in zip(columns, metadata["columns"])]
                    grouped.setdefault(table, []).append((row, columns, values))
                except Exception as exc:
                    yield pvalue.TaggedOutput("errors", json.dumps({"stage": "decode", "error": str(exc), "row": record}, ensure_ascii=False))
            for table, rows in grouped.items():
                for attempt in range(4):
                    try:
                        with self.database.batch() as batch:
                            batch.insert_or_update(table, columns=rows[0][1], values=[item[2] for item in rows])
                        self.ok.inc(len(rows))
                        break
                    except Exception as exc:
                        if attempt < 3:
                            time.sleep(2 ** attempt)
                        else:
                            for original, _, _ in rows:
                                self.failed.inc()
                                yield pvalue.TaggedOutput("errors", json.dumps({"stage": "write", "table": table, "error": str(exc), "row": original}, ensure_ascii=False))

    opts = ["--runner=DataflowRunner", f"--project={env('GOOGLE_CLOUD_PROJECT')}", f"--region={env('DATAFLOW_REGION')}",
            f"--temp_location={env('DATAFLOW_TEMP_GCS')}", f"--staging_location={env('DATAFLOW_STAGING_GCS')}",
            f"--num_workers={os.getenv('DATAFLOW_NUM_WORKERS', '8')}", f"--max_num_workers={os.getenv('DATAFLOW_MAX_WORKERS', '50')}",
            f"--machine_type={os.getenv('DATAFLOW_MACHINE_TYPE', 'n2-standard-4')}", "--autoscaling_algorithm=THROUGHPUT_BASED"]
    worker_sa = os.getenv("DATAFLOW_WORKER_SERVICE_ACCOUNT")
    if worker_sa: opts.append(f"--service_account_email={worker_sa}")
    options = PipelineOptions(opts)
    setup = options.view_as(SetupOptions)
    setup.requirements_file = str(Path(__file__).with_name("requirements-dataflow.txt"))
    setup.save_main_session = True
    pipeline = beam.Pipeline(options=options)
    output = (pipeline
              | "ReadAuroraGCSFiles" >> beam.io.ReadFromText(f"gs://{bucket}/{prefix}/data/*/*.jsonl.gz", compression_type=CompressionTypes.AUTO)
              | "BatchRows" >> BatchElements(min_batch_size=20, max_batch_size=int(os.getenv("SPANNER_BATCH_ROWS", "50")))
              | "WriteSpanner" >> beam.ParDo(SpannerWriter(env("GOOGLE_CLOUD_PROJECT"), env("SPANNER_INSTANCE_ID"), env("SPANNER_DATABASE_ID"), table_schemas)).with_outputs("errors"))
    output.errors | "WriteSpannerDeadLetters" >> beam.io.WriteToText(env("SPANNER_DEAD_LETTER_PREFIX").rstrip("/") + "/errors", file_name_suffix=".jsonl")
    result = pipeline.run()
    state = result.wait_until_finish()
    if str(state) != "DONE":
        raise RuntimeError(f"Dataflow job ended in state {state}")
    metrics = result.metrics().query().get("counters", [])
    written = sum(int(m.committed if m.committed is not None else m.attempted or 0) for m in metrics if m.key.metric.name == "rows_written")
    gcs.bucket(bucket).blob(f"{prefix}/dataflow-result.json").upload_from_string(json.dumps({"job_id": result.job_id(), "state": str(state), "rows_written": written}, indent=2), content_type="application/json")
    print(json.dumps({"job_id": result.job_id(), "state": str(state), "rows_written": written}))


def validate() -> None:
    gcs, bucket, prefix = gcs_parts()
    manifest = json.loads(gcs.bucket(bucket).blob(f"{prefix}/export-manifest.json").download_as_text())
    result = json.loads(gcs.bucket(bucket).blob(f"{prefix}/dataflow-result.json").download_as_text())
    expected = sum(int(v) for v in manifest.values())
    uri = env("SPANNER_DEAD_LETTER_PREFIX")
    base = f"gs://{bucket}/"
    if not uri.startswith(base): raise ValueError("SPANNER_DEAD_LETTER_PREFIX must be inside GCS_BUCKET")
    errors = [b for b in gcs.list_blobs(bucket, prefix=uri.removeprefix(base).rstrip("/") + "/") if (b.size or 0) > 0]
    written = int(result["rows_written"])
    report = {"exported_rows": expected, "spanner_rows_written": written, "counts_match": expected == written, "dead_letter_objects": len(errors), "dataflow_job_id": result["job_id"]}
    if expected != written or errors:
        raise RuntimeError("Validation failed: " + json.dumps(report))
    gcs.bucket(bucket).blob(f"{prefix}/validation-report.json").upload_from_string(json.dumps(report, indent=2), content_type="application/json")
    print(json.dumps({"validation": "passed", **report}))
