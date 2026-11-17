import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import google.auth
from google.auth.transport.requests import AuthorizedSession
from dotenv import load_dotenv

API = "https://storagetransfer.googleapis.com/v1"


def main():
    load_dotenv()
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    gcs_bucket = os.environ["GCS_TRANSFER_BUCKET"]
    gcs_prefix = os.getenv("GCS_TRANSFER_PREFIX", "dynamodb-imports").strip("/")
    metadata_path = Path(os.getenv("EXPORT_METADATA_FILE", "dynamodb_export_metadata.json"))
    if not metadata_path.exists():
        raise SystemExit(f"Export metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("transfer_status") == "SUCCESS" and metadata.get("gcs_input_pattern"):
        print("Reusing completed Storage Transfer operation.")
        print(f"Dataflow input pattern: {metadata['gcs_input_pattern']}")
        return
    role_arn = os.environ["AWS_TRANSFER_ROLE_ARN"]
    export_id = metadata["export_arn"].rsplit("/", 1)[-1]
    source_path = metadata["s3_data_prefix"].strip("/")
    destination_prefix = f"{gcs_prefix}/{metadata['table_name']}/{export_id}".strip("/")
    job_slug = re.sub(r"[^A-Za-z0-9._~-]", "-", f"ddb-{metadata['table_name']}-{export_id[-12:]}")
    transfer_spec = {
        "awsS3DataSource": {
            "bucketName": metadata["s3_bucket"],
            "path": source_path,
            "roleArn": role_arn,
        },
        "gcsDataSink": {"bucketName": gcs_bucket, "path": destination_prefix},
        "transferOptions": {"overwriteWhen": "DIFFERENT"},
    }
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    job_name = metadata.get("transfer_job_name")
    if job_name:
        get_job = session.get(f"{API}/{job_name}", params={"projectId": project}, timeout=60)
        get_job.raise_for_status()
        print(f"Resuming Storage Transfer job {job_name}")
    else:
        create = session.post(f"{API}/transferJobs", json={
            "projectId": project,
            "name": "transferJobs/" + job_slug,
            "description": "DynamoDB export copy " + export_id,
            "status": "ENABLED",
            "transferSpec": transfer_spec,
        }, timeout=60)
        create.raise_for_status()
        job_name = create.json()["name"]
        metadata["transfer_job_name"] = job_name
        metadata["transfer_status"] = "IN_PROGRESS"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Created Storage Transfer job {job_name}")

    run = session.post(f"{API}/{job_name}:run", json={"projectId": project}, timeout=60)
    run.raise_for_status()
    operation_name = run.json()["name"]
    metadata["transfer_operation_name"] = operation_name
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Started transfer operation {operation_name}")
    deadline = time.time() + int(os.getenv("TRANSFER_TIMEOUT_HOURS", "48")) * 3600
    while True:
        response = session.get(f"{API}/{quote(operation_name, safe='/')}", timeout=60)
        response.raise_for_status()
        operation = response.json()
        metadata_op = operation.get("metadata", {})
        status = metadata_op.get("status", "IN_PROGRESS")
        counters = metadata_op.get("counters", {})
        print(f"Storage Transfer status: {status}; bytes copied: {counters.get('bytesCopiedToSink', 0)}")
        if operation.get("done"):
            if operation.get("error"):
                raise RuntimeError(f"Storage Transfer failed: {operation['error']}")
            if status not in ("SUCCESS", "STATUS_UNSPECIFIED"):
                raise RuntimeError(f"Storage Transfer ended with status {status}")
            break
        if time.time() >= deadline:
            raise TimeoutError("Storage Transfer did not finish before TRANSFER_TIMEOUT_HOURS")
        time.sleep(30)

    destination_uri = f"gs://{gcs_bucket}/{destination_prefix}/"
    metadata["gcs_data_prefix"] = destination_uri
    metadata["gcs_input_pattern"] = destination_uri + "*.json.gz"
    metadata["transfer_job_name"] = job_name
    metadata["transfer_operation_name"] = operation_name
    metadata["transfer_counters"] = counters
    metadata["transfer_status"] = "SUCCESS"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Transfer complete. Dataflow input: {metadata['gcs_input_pattern']}")


if __name__ == "__main__":
    main()
