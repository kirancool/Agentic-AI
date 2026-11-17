import json
import os
import re
import time
import urllib.parse
from collections import defaultdict

import mysql.connector
from dotenv import load_dotenv
import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import spanner
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

MYSQL_CFG = {
    "host": os.environ["MYSQL_HOST"],
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.environ["MYSQL_USER"],
    "password": os.environ["MYSQL_PASSWORD"],
    "database": os.environ["MYSQL_DATABASE"],
}
SPANNER_INSTANCE = os.environ["SPANNER_INSTANCE"]
SPANNER_DATABASE = os.environ["SPANNER_DATABASE"]
SPANNER_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
DATAFLOW_REGION = os.environ["DATAFLOW_REGION"]
DATAFLOW_OUTPUT = os.environ["DATAFLOW_OUTPUT_GCS"]
DATAFLOW_TEMP = os.environ["DATAFLOW_TEMP_GCS"]
DATAFLOW_STAGING = os.environ["DATAFLOW_STAGING_GCS"]
DATAFLOW_SUBNETWORK = os.getenv("DATAFLOW_SUBNETWORK")
DATAFLOW_SERVICE_ACCOUNT = os.getenv("DATAFLOW_SERVICE_ACCOUNT")
DATAFLOW_MAX_WORKERS = int(os.getenv("DATAFLOW_MAX_WORKERS", "10"))
DATAFLOW_TEMPLATE_VERSION = os.getenv("DATAFLOW_TEMPLATE_VERSION", "latest")
spanner_client = spanner.Client(project=SPANNER_PROJECT)
database = spanner_client.instance(SPANNER_INSTANCE).database(SPANNER_DATABASE)
llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=1,
    top_p=0.95,
    max_completion_tokens=4096,
    timeout=300.0,
)


def mysql_connection():
    return mysql.connector.connect(**MYSQL_CFG)


def qi(name: str) -> str:
    """Quote a GoogleSQL identifier (table/column names)."""
    return "`" + name.replace("`", "``") + "`"


def mysql_qi(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def spanner_type(mysql_type: str) -> str:
    """Best-effort MySQL-to-Spanner GoogleSQL type conversion; review before production."""
    t = mysql_type.lower().strip()
    unsigned = "unsigned" in t
    if t.startswith("tinyint(1)"):
        return "BOOL"
    if t.startswith(("tinyint", "smallint", "mediumint", "int", "integer", "bigint")):
        # Spanner INT64 is signed. Unsigned BIGINT needs NUMERIC to preserve its full range.
        if t.startswith("bigint") and unsigned:
            return "NUMERIC"
        return "INT64"
    if t.startswith(("decimal", "numeric")):
        match = re.search(r"\((\d+)(?:,(\d+))?\)", t)
        precision = int(match.group(1)) if match else 38
        scale = int(match.group(2) or 0) if match else 9
        if precision > 38 or scale > 9:
            return "STRING(MAX)"  # MySQL precision exceeds Spanner NUMERIC limits; preserve text.
        return "NUMERIC"
    if t.startswith(("float", "double", "real")):
        return "FLOAT64"
    if t.startswith(("varchar", "char", "enum", "set")):
        match = re.search(r"\((\d+)\)", t)
        return f"STRING({match.group(1)})" if match else "STRING(MAX)"
    if "text" in t or t in ("json", "longtext", "mediumtext"):
        return "JSON" if t == "json" else "STRING(MAX)"
    if "blob" in t or "binary" in t:
        match = re.search(r"\((\d+)\)", t)
        return f"BYTES({match.group(1)})" if match else "BYTES(MAX)"
    if t == "date":
        return "DATE"
    if t.startswith(("datetime", "timestamp")):
        return "TIMESTAMP"
    if t.startswith("time"):
        return "STRING(32)"  # Spanner has no TIME type.
    if t.startswith("year"):
        return "INT64"
    return "STRING(MAX)"


def inspect_mysql_source() -> str:
    """Inventory source tables, columns, indexes, foreign keys and program objects."""
    conn = mysql_connection()
    cur = conn.cursor(dictionary=True)
    db = MYSQL_CFG["database"]
    try:
        cur.execute("""SELECT TABLE_NAME,TABLE_TYPE,ENGINE,TABLE_ROWS
                       FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s""", (db,))
        tables = cur.fetchall()
        cur.execute("""SELECT TABLE_NAME,COLUMN_NAME,ORDINAL_POSITION,COLUMN_TYPE,
                    IS_NULLABLE,COLUMN_DEFAULT,EXTRA,COLUMN_COMMENT
                    FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s
                    ORDER BY TABLE_NAME,ORDINAL_POSITION""", (db,))
        columns = cur.fetchall()
        cur.execute("""SELECT TABLE_NAME,INDEX_NAME,NON_UNIQUE,SEQ_IN_INDEX,COLUMN_NAME
                    FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=%s
                    ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX""", (db,))
        indexes = cur.fetchall()
        cur.execute("""SELECT TABLE_NAME,CONSTRAINT_NAME,COLUMN_NAME,
                    REFERENCED_TABLE_NAME,REFERENCED_COLUMN_NAME,ORDINAL_POSITION
                    FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA=%s
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                    ORDER BY TABLE_NAME,CONSTRAINT_NAME,ORDINAL_POSITION""", (db,))
        foreign_keys = cur.fetchall()
        cur.execute("SELECT TABLE_NAME,VIEW_DEFINITION FROM information_schema.VIEWS WHERE TABLE_SCHEMA=%s", (db,))
        views = cur.fetchall()
        cur.execute("SELECT ROUTINE_TYPE,ROUTINE_NAME,ROUTINE_DEFINITION FROM information_schema.ROUTINES WHERE ROUTINE_SCHEMA=%s", (db,))
        routines = cur.fetchall()
        cur.execute("""SELECT TRIGGER_NAME,ACTION_TIMING,EVENT_MANIPULATION,
                    EVENT_OBJECT_TABLE,ACTION_STATEMENT FROM information_schema.TRIGGERS
                    WHERE TRIGGER_SCHEMA=%s""", (db,))
        triggers = cur.fetchall()
        counts = {}
        for row in tables:
            if row["TABLE_TYPE"] == "BASE TABLE":
                cur.execute(f"SELECT COUNT(*) AS n FROM {mysql_qi(row['TABLE_NAME'])}")
                counts[row["TABLE_NAME"]] = cur.fetchone()["n"]
        return json.dumps({"tables": tables, "columns": columns, "indexes": indexes,
                           "foreign_keys": foreign_keys, "views": views, "routines": routines,
                           "triggers": triggers, "exact_row_counts": counts}, default=str)
    finally:
        cur.close()
        conn.close()


def create_spanner_schema() -> str:
    """Create GoogleSQL base tables, then unique/secondary indexes and foreign keys."""
    conn = mysql_connection()
    cur = conn.cursor(dictionary=True)
    db = MYSQL_CFG["database"]
    try:
        cur.execute("""SELECT TABLE_NAME,COLUMN_NAME,ORDINAL_POSITION,COLUMN_TYPE,
                    IS_NULLABLE,COLUMN_DEFAULT,EXTRA FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME,ORDINAL_POSITION""", (db,))
        columns = cur.fetchall()
        cur.execute("""SELECT TABLE_NAME,INDEX_NAME,NON_UNIQUE,SEQ_IN_INDEX,COLUMN_NAME
                    FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=%s
                    ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX""", (db,))
        indexes = cur.fetchall()
        cur.execute("""SELECT TABLE_NAME,CONSTRAINT_NAME,COLUMN_NAME,REFERENCED_TABLE_NAME,
                    REFERENCED_COLUMN_NAME,ORDINAL_POSITION FROM information_schema.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA=%s AND REFERENCED_TABLE_NAME IS NOT NULL
                    ORDER BY TABLE_NAME,CONSTRAINT_NAME,ORDINAL_POSITION""", (db,))
        fks = cur.fetchall()

        grouped = defaultdict(list)
        for col in columns:
            grouped[col["TABLE_NAME"]].append(col)
        idx_groups = defaultdict(list)
        for idx in indexes:
            idx_groups[(idx["TABLE_NAME"], idx["INDEX_NAME"])].append(idx)

        ddl = []
        for table, cols in grouped.items():
            primary = [x["COLUMN_NAME"] for x in idx_groups.get((table, "PRIMARY"), [])]
            if not primary:
                raise ValueError(f"Table {table!r} has no primary key; Spanner requires a primary key. Design one manually.")
            defs = []
            for col in cols:
                name = qi(col["COLUMN_NAME"])
                typ = spanner_type(col["COLUMN_TYPE"])
                null = " NOT NULL" if col["IS_NULLABLE"] == "NO" else ""
                # Column defaults / AUTO_INCREMENT require application-specific strategy in Spanner.
                defs.append(f"  {name} {typ}{null}")
            pk = ", ".join(qi(c) for c in primary)
            ddl.append(f"CREATE TABLE {qi(table)} (\n" + ",\n".join(defs) + f"\n) PRIMARY KEY ({pk})")

        # Index names must be unique in the database; retain source table and index for clarity.
        for (table, name), parts in idx_groups.items():
            if name == "PRIMARY":
                continue
            cols_sql = ", ".join(qi(x["COLUMN_NAME"]) for x in parts)
            idx_name = re.sub(r"[^A-Za-z0-9_]", "_", f"{table}_{name}")[:128]
            unique = "UNIQUE " if not parts[0]["NON_UNIQUE"] else ""
            ddl.append(f"CREATE {unique}INDEX {qi(idx_name)} ON {qi(table)} ({cols_sql})")

        fk_groups = defaultdict(list)
        for fk in fks:
            fk_groups[(fk["TABLE_NAME"], fk["CONSTRAINT_NAME"], fk["REFERENCED_TABLE_NAME"])].append(fk)
        for (table, constraint, ref_table), parts in fk_groups.items():
            lhs = ", ".join(qi(p["COLUMN_NAME"]) for p in parts)
            rhs = ", ".join(qi(p["REFERENCED_COLUMN_NAME"]) for p in parts)
            fk_name = re.sub(r"[^A-Za-z0-9_]", "_", constraint)[:128]
            ddl.append(f"ALTER TABLE {qi(table)} ADD CONSTRAINT {qi(fk_name)} FOREIGN KEY ({lhs}) REFERENCES {qi(ref_table)} ({rhs})")

        # Spanner DDL is applied asynchronously and is not transactional. Stop and report on the first error.
        applied = 0
        for statement in ddl:
            operation = database.update_ddl([statement])
            operation.result()  # Wait for this statement before applying dependent DDL.
            applied += 1
        return f"Applied {applied} Spanner DDL statement(s) for {len(grouped)} tables. Review converted types, defaults, keys and indexes before loading."
    except Exception as exc:
        return f"Schema creation stopped after partial DDL application: {exc}"
    finally:
        cur.close()
        conn.close()


def launch_dataflow_migration(table_names: list[str]) -> dict:
    """Launch the Google SourceDB-to-Spanner batch template and wait for completion."""
    if not SPANNER_PROJECT:
        raise ValueError("GOOGLE_CLOUD_PROJECT is required")
    if not DATAFLOW_OUTPUT.startswith("gs://") or not DATAFLOW_TEMP.startswith("gs://") or not DATAFLOW_STAGING.startswith("gs://"):
        raise ValueError("DATAFLOW_OUTPUT_GCS, DATAFLOW_TEMP_GCS, and DATAFLOW_STAGING_GCS must be gs:// paths")

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(credentials)
    job_name = "aurora-mysql-to-spanner-" + str(int(time.time()))
    jdbc_url = "jdbc:mysql://{}:{}/{}".format(
        MYSQL_CFG["host"], MYSQL_CFG["port"], urllib.parse.quote(MYSQL_CFG["database"])
    )
    parameters = {
        "sourceConfigURL": jdbc_url,
        "sourceDbDialect": "MYSQL",
        "username": MYSQL_CFG["user"],
        "password": MYSQL_CFG["password"],
        "instanceId": SPANNER_INSTANCE,
        "databaseId": SPANNER_DATABASE,
        "projectId": SPANNER_PROJECT,
        "outputDirectory": DATAFLOW_OUTPUT,
        "tables": ",".join(table_names),
    }
    template_path = (
        f"gs://dataflow-templates-{DATAFLOW_REGION}/{DATAFLOW_TEMPLATE_VERSION}"
        "/flex/Sourcedb_to_Spanner_Flex"
    )
    environment = {
        "tempLocation": DATAFLOW_TEMP,
        "stagingLocation": DATAFLOW_STAGING,
        "maxWorkers": DATAFLOW_MAX_WORKERS,
    }
    if DATAFLOW_SUBNETWORK:
        environment["subnetwork"] = DATAFLOW_SUBNETWORK
    if DATAFLOW_SERVICE_ACCOUNT:
        environment["serviceAccountEmail"] = DATAFLOW_SERVICE_ACCOUNT

    launch_url = (
        f"https://dataflow.googleapis.com/v1b3/projects/{SPANNER_PROJECT}"
        f"/locations/{DATAFLOW_REGION}/flexTemplates:launch"
    )
    response = session.post(launch_url, json={"launchParameter": {
        "jobName": job_name,
        "parameters": parameters,
        "containerSpecGcsPath": template_path,
        "environment": environment,
    }}, timeout=60)
    response.raise_for_status()
    launch = response.json()
    job = launch.get("job", {})
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"Dataflow launch response did not include a job ID: {launch}")
    print(f"Started Dataflow job {job.get('name', job_name)} ({job_id})")

    job_url = (
        f"https://dataflow.googleapis.com/v1b3/projects/{SPANNER_PROJECT}"
        f"/locations/{DATAFLOW_REGION}/jobs/{job_id}"
    )
    terminal = {"JOB_STATE_DONE", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_DRAINED", "JOB_STATE_UPDATED"}
    while True:
        status_response = session.get(job_url, timeout=60)
        status_response.raise_for_status()
        state = status_response.json().get("currentState", "UNKNOWN")
        print(f"Dataflow job state: {state}")
        if state in terminal:
            if state != "JOB_STATE_DONE":
                raise RuntimeError(f"Dataflow job ended with state {state}; inspect job logs and {DATAFLOW_OUTPUT}")
            return {"job_id": job_id, "job_name": job.get("name", job_name), "state": state}
        time.sleep(30)


def validate_migration(table_names: list[str]) -> str:
    """Compare exact row counts for source and Spanner tables."""
    conn = mysql_connection()
    cur = conn.cursor()
    report = []
    try:
        for name in table_names:
            cur.execute(f"SELECT COUNT(*) FROM {mysql_qi(name)}")
            source_count = cur.fetchone()[0]
            rows = list(database.execute_sql(f"SELECT COUNT(*) FROM {qi(name)}"))
            target_count = rows[0][0]
            report.append({"table": name, "mysql": source_count, "spanner": target_count,
                           "match": source_count == target_count})
        return json.dumps(report)
    except Exception as exc:
        return f"Validation failed: {exc}"
    finally:
        cur.close()
        conn.close()


def main():
    print("Inspecting Aurora MySQL source...")
    source_report = inspect_mysql_source()
    inventory = json.loads(source_report)
    print(json.dumps(inventory, indent=2, default=str))
    names = [t["TABLE_NAME"] for t in inventory["tables"] if t["TABLE_TYPE"] == "BASE TABLE"]
    if not names:
        raise SystemExit("No base tables found.")

    print("\n=== NEMOTRON MIGRATION ASSESSMENT ===\n")
    assessment = llm.invoke(
        "Review this Aurora MySQL to Google Cloud Spanner migration inventory. "
        "Summarize high-risk type mappings, primary key and index concerns, "
        "unsupported database objects, and likely migration risks. Do not claim "
        "schema or data was migrated. Inventory:\n" + source_report
    )
    print(assessment.content)

    print("\nCreating Spanner schema...")
    result = create_spanner_schema()
    print(result)
    if "stopped" in result.lower():
        raise SystemExit("Schema creation was incomplete; review errors before continuing.")

    print("\nLaunching Dataflow batch migration...")
    job = launch_dataflow_migration(names)
    print(json.dumps(job, indent=2))

    print("\nValidating row counts...")
    result = validate_migration(names)
    print(result)
    if result.startswith("Validation failed"):
        raise SystemExit(result)
    checks = json.loads(result)
    if any(not x["match"] for x in checks):
        raise SystemExit("Source/Spanner row counts differ.")
    print("Base-table copy completed and row counts match.")
    print("Views, routines, triggers, defaults, enum semantics, and application ID generation require separate review.")


if __name__ == "__main__":
    main()
