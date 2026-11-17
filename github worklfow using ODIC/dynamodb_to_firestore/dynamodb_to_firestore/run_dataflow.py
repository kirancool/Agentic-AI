import json
import os
import re
import subprocess
import sys
from pathlib import Path

import google.auth
from google.auth.transport.requests import AuthorizedSession
from dotenv import load_dotenv


def fetch_dataflow_metrics(project, region, job_id):
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    url = f"https://dataflow.googleapis.com/v1b3/projects/{project}/locations/{region}/jobs/{job_id}/metrics"
    response = session.get(url, timeout=60)
    response.raise_for_status()
    totals = {}
    for update in response.json().get("metrics", []):
        metric_name = update.get("name", {}).get("name", "")
        short_name = metric_name.rsplit("/", 1)[-1]
        if short_name not in {"items_queued", "items_succeeded", "parse_failures", "write_failures"}:
            continue
        value = update.get("scalar", update.get("tentative", 0))
        if isinstance(value, dict):
            value = value.get("int64Value", value.get("value", 0))
        try:
            totals[short_name] = max(totals.get(short_name, 0), int(value))
        except (TypeError, ValueError):
            continue
    return totals


def main():
    load_dotenv()
    metadata_path = Path(os.getenv("EXPORT_METADATA_FILE", "dynamodb_export_metadata.json"))
    if not metadata_path.exists():
        raise SystemExit(f"Export metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_pattern = metadata.get("gcs_input_pattern")
    if not input_pattern:
        raise SystemExit("Complete transfer_s3_to_gcs.py before launching Dataflow.")

    schema = metadata["key_schema"]
    partition_key = os.getenv("DDB_PARTITION_KEY") or next(x["AttributeName"] for x in schema if x["KeyType"] == "HASH")
    sort_key = os.getenv("DDB_SORT_KEY")
    if sort_key is None:
        sort_key = next((x["AttributeName"] for x in schema if x["KeyType"] == "RANGE"), "")
    export_id = metadata["export_arn"].rsplit("/", 1)[-1]
    dead_letter_prefix = os.environ["DDB_DEAD_LETTER_GCS"].rstrip("/") + "/" + export_id

    args = [
        sys.executable, "-u", str(Path(__file__).with_name("dynamodb_export_to_firestore.py")),
        "--input-pattern", input_pattern,
        "--collection", os.environ.get("FIRESTORE_COLLECTION", metadata["table_name"]),
        "--firestore-database", os.getenv("FIRESTORE_DATABASE", "(default)"),
        "--partition-key", partition_key,
        "--sort-key", sort_key,
        "--dead-letter-prefix", dead_letter_prefix,
        "--retry-limit", os.getenv("FIRESTORE_RETRY_LIMIT", "5"),
        "--initial-write-rate", os.getenv("FIRESTORE_INITIAL_WRITE_RATE", "50"),
        "--max-write-rate", os.getenv("FIRESTORE_MAX_WRITE_RATE", "500"),
        "--requirements-file", str(Path(__file__).with_name("requirements-dataflow.txt")),
        "--runner=DataflowRunner",
        f"--project={os.environ['GOOGLE_CLOUD_PROJECT']}",
        f"--region={os.environ['DATAFLOW_REGION']}",
        f"--temp_location={os.environ['DATAFLOW_TEMP_GCS']}",
        f"--staging_location={os.environ['DATAFLOW_STAGING_GCS']}",
        f"--service_account_email={os.environ['DATAFLOW_WORKER_SERVICE_ACCOUNT']}",
        f"--machine_type={os.getenv('DATAFLOW_MACHINE_TYPE', 'n2-standard-4')}",
        f"--num_workers={os.getenv('DATAFLOW_NUM_WORKERS', '8')}",
        f"--max_num_workers={os.getenv('DATAFLOW_MAX_WORKERS', '50')}",
        f"--autoscaling_algorithm={os.getenv('DATAFLOW_AUTOSCALING_ALGORITHM', 'THROUGHPUT_BASED')}",
        f"--disk_size_gb={os.getenv('DATAFLOW_DISK_SIZE_GB', '100')}",
        "--experiments=use_runner_v2",
    ]
    print("Launching Dataflow Python migration job...", flush=True)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    job_id = None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = re.search(r"DATAFLOW_JOB_ID=([A-Za-z0-9_-]+)", line)
        if match:
            job_id = match.group(1)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, args)
    if not job_id:
        raise RuntimeError("Dataflow process ended but did not report a job ID")

    metrics_error = None
    try:
        metrics = fetch_dataflow_metrics(os.environ["GOOGLE_CLOUD_PROJECT"], os.environ["DATAFLOW_REGION"], job_id)
    except Exception as exc:
        metrics = {}
        metrics_error = str(exc)
    metadata["dataflow_job_id"] = job_id
    metadata["dataflow_final_state"] = "DONE"
    metadata["dataflow_metrics"] = metrics
    metadata["dataflow_metrics_error"] = metrics_error
    metadata["dead_letter_prefix"] = dead_letter_prefix
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"job_id": job_id, "state": "DONE", "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
