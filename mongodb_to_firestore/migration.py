"""Python-only Amazon DocumentDB -> GCS -> Dataflow -> Firestore migration."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import tempfile
from pathlib import Path

from bson import json_util
from google.cloud import firestore, storage
from pymongo import MongoClient


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ValueError(f"Required environment variable is missing: {name}")
    return value


def mongo_client() -> MongoClient:
    host = env("DOCDB_HOST")
    port = os.getenv("DOCDB_PORT", "27017")
    ca_file = env("DOCDB_CA_FILE", "/tmp/global-bundle.pem")
    auth_mode = os.getenv("DOCDB_AUTH_MODE", "iam").lower()
    query = (
        f"tls=true&tlsCAFile={ca_file}&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false"
    )
    if auth_mode == "iam":
        query += "&authSource=%24external&authMechanism=MONGODB-AWS"
    else:
        raise ValueError("This workflow requires DOCDB_AUTH_MODE=iam; password-based DOCDB_URI auth is not configured")
    uri = f"mongodb://{host}:{port}/?{query}"
    return MongoClient(uri, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)


def gcs_parts() -> tuple[storage.Client, str, str]:
    return storage.Client(project=env("GOOGLE_CLOUD_PROJECT")), env("GCS_BUCKET"), env("GCS_PREFIX").strip("/")


def preflight() -> None:
    client = mongo_client()
    try:
        client.admin.command("ping")
        db = client[env("DOCDB_DATABASE")]
        names = db.list_collection_names()
        if not names:
            raise RuntimeError("The selected DocumentDB database has no collections")
        gcs, bucket_name, _ = gcs_parts()
        bucket = gcs.lookup_bucket(bucket_name)
        if bucket is None:
            raise RuntimeError(f"GCS bucket does not exist or is inaccessible: {bucket_name}")
        firestore.Client(project=env("GOOGLE_CLOUD_PROJECT"), database=env("FIRESTORE_DATABASE", "(default)")).get_all([])
        print(json.dumps({"preflight": "ok", "collections": names, "gcs_bucket": bucket_name}))
    finally:
        client.close()


def ai_review() -> None:
    """Ask Astra to review only collection/index metadata, never document values."""
    from openai import OpenAI

    client = mongo_client()
    try:
        database = client[env("DOCDB_DATABASE")]
        schema = []
        for collection_name in database.list_collection_names():
            collection = database[collection_name]
            indexes = collection.index_information()
            schema.append({
                "collection": collection_name,
                "estimated_documents": collection.estimated_document_count(),
                "indexes": [
                    {"name": index_name, "keys": index.get("key", []), "unique": bool(index.get("unique", False)), "expire_after_seconds": index.get("expireAfterSeconds")}
                    for index_name, index in indexes.items()
                ],
            })
    finally:
        client.close()

    response = OpenAI(api_key=env("OPENAI_API_KEY")).responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-astra"),
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": "You are the schema-review agent for an automated Amazon DocumentDB to Firestore migration. Assess migration risks from collection names, estimated counts, and index metadata only. Never claim data or indexes are migrated. Flag Firestore modeling, TTL, index/query redesign, oversized/large-collection and cutover risks. Do not instruct the workflow to modify or delete data. Return concise actionable review findings."},
            {"role": "user", "content": "Review this metadata-only source inventory for a MongoDB-compatible DocumentDB to Firestore migration. Return JSON per the requested schema. Inventory: " + json.dumps(schema, ensure_ascii=False)},
        ],
        text={"format": {
            "type": "json_schema",
            "name": "documentdb_firestore_review",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "recommended_checks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "risks", "recommended_checks"],
                "additionalProperties": False,
            },
        }},
    )
    report = json.loads(response.output_text)
    report["model"] = os.getenv("OPENAI_MODEL", "gpt-6-astra")
    report["review_scope"] = "collection names, estimated counts, and index metadata; no document values"
    gcs, bucket_name, prefix = gcs_parts()
    blob_name = f"{prefix}/ai-review.json"
    gcs.bucket(bucket_name).blob(blob_name).upload_from_string(json.dumps(report, indent=2), content_type="application/json")
    print(json.dumps({"ai_review": "complete", "model": report["model"], "report": f"gs://{bucket_name}/{blob_name}", "risks": len(report["risks"])}))


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "collection"
    if cleaned.startswith("__") and cleaned.endswith("__"):
        cleaned = "c_" + cleaned.strip("_")
    if cleaned != name:
        import hashlib
        cleaned = f"{cleaned[:180]}_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:12]}"
    return cleaned[:200]


def export_to_gcs() -> None:
    client = mongo_client()
    gcs, bucket_name, prefix = gcs_parts()
    bucket = gcs.bucket(bucket_name)
    db = client[env("DOCDB_DATABASE")]
    chunk_bytes = int(os.getenv("MONGO_EXPORT_CHUNK_MB", "128")) * 1024 * 1024
    batch_size = int(os.getenv("MONGO_EXPORT_BATCH_SIZE", "1000"))
    manifest: dict[str, int] = {}
    try:
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            count = 0
            part = 0
            temporary: Path | None = None
            stream = None

            def open_chunk() -> tuple[Path, object]:
                temp = tempfile.NamedTemporaryFile(prefix="mongo-export-", suffix=".jsonl.gz", delete=False)
                temp.close()
                path = Path(temp.name)
                return path, gzip.open(path, "wt", encoding="utf-8")

            def upload_chunk(path: Path, index: int) -> None:
                blob_name = f"{prefix}/data/{safe_name(collection_name)}/part-{index:08d}.jsonl.gz"
                bucket.blob(blob_name).upload_from_filename(str(path), content_type="application/gzip")
                path.unlink(missing_ok=True)
                print(f"Uploaded gs://{bucket_name}/{blob_name}", flush=True)

            try:
                for document in collection.find({}).batch_size(batch_size):
                    if stream is None:
                        temporary, stream = open_chunk()
                    record = {"collection": collection_name, "document": json_util.dumps(document, json_options=json_util.CANONICAL_JSON_OPTIONS)}
                    stream.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                    count += 1
                    stream.flush()
                    if temporary and temporary.stat().st_size >= chunk_bytes:
                        stream.close()
                        upload_chunk(temporary, part)
                        part += 1
                        temporary, stream = None, None
                if stream is not None and temporary is not None:
                    stream.close()
                    upload_chunk(temporary, part)
            finally:
                if stream is not None:
                    stream.close()
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            manifest[collection_name] = count
            print(f"Exported {count} documents from {collection_name}", flush=True)
        manifest_blob = f"{prefix}/export-manifest.json"
        bucket.blob(manifest_blob).upload_from_string(json.dumps(manifest, indent=2), content_type="application/json")
        print(json.dumps({"exported_total": sum(manifest.values()), "manifest": f"gs://{bucket_name}/{manifest_blob}"}))
    finally:
        client.close()


def load_to_firestore() -> None:
    # Imports here ensure GitHub's controller process does not need to initialize Beam before options.
    import apache_beam as beam
    from apache_beam.io.filesystem import CompressionTypes
    from apache_beam.metrics.metric import Metrics
    from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
    from apache_beam import pvalue
    from apache_beam.transforms.window import GlobalWindow
    from apache_beam.utils.timestamp import MAX_TIMESTAMP
    from apache_beam.utils.windowed_value import WindowedValue
    from queue import Empty, Queue
    from bson import json_util
    from bson.binary import Binary
    from bson.decimal128 import Decimal128
    from bson.objectid import ObjectId
    from bson.regex import Regex
    from datetime import datetime, timezone
    import math
    import hashlib
    import base64

    def encode_value(value):
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, Decimal128):
            return str(value)
        if isinstance(value, Binary):
            return bytes(value)
        if isinstance(value, Regex):
            return {"pattern": value.pattern, "options": value.flags}
        if isinstance(value, datetime):
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
        if isinstance(value, dict):
            converted = {}
            for key, nested in value.items():
                if key.startswith("__") and key.endswith("__"):
                    raise ValueError(f"Firestore-reserved field name: {key}")
                converted[key] = encode_value(nested)
            return converted
        if isinstance(value, list):
            return [{"mongo_type": "array", "items": encode_value(item)} if isinstance(item, list) else encode_value(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("NaN and Infinity are not supported Firestore values")
        if isinstance(value, int) and not -(2**63) <= value < 2**63:
            return str(value)
        if value is None or isinstance(value, (str, bool, int, float, bytes)):
            return value
        return str(value)

    class FirestoreWriter(beam.DoFn):
        def __init__(self, project, database, retry_limit, collection_prefix):
            self.project = project
            self.database = database
            self.retry_limit = retry_limit
            self.collection_prefix = collection_prefix
            self.success = Metrics.counter("mongo_firestore", "items_succeeded")
            self.parse_fail = Metrics.counter("mongo_firestore", "parse_failures")
            self.write_fail = Metrics.counter("mongo_firestore", "write_failures")

        def setup(self):
            self.client = firestore.Client(project=self.project, database=self.database)

        def start_bundle(self):
            self.errors = Queue()
            self.writer = self.client.bulk_writer()
            self.writer.on_write_result(lambda reference, result, writer: self.success.inc())
            self.writer.on_write_error(self.on_write_error)

        def on_write_error(self, failure, writer):
            if failure.attempts < self.retry_limit:
                return True
            self.errors.put(json.dumps({"path": getattr(getattr(failure.operation, "reference", None), "path", "unknown"), "code": str(failure.code), "error": failure.message, "attempts": failure.attempts}))
            self.write_fail.inc()
            return False

        def process(self, line):
            try:
                row = json.loads(line)
                source_collection = row["collection"]
                document = json_util.loads(row["document"])
                if "_id" not in document:
                    raise ValueError("MongoDB document is missing _id")
                original_id = document.pop("_id")
                id_json = json_util.dumps(original_id, json_options=json_util.CANONICAL_JSON_OPTIONS, sort_keys=True)
                doc_id = "k_" + base64.urlsafe_b64encode(id_json.encode()).decode().rstrip("=")
                if len(doc_id.encode("utf-8")) > 1400:
                    doc_id = "h_" + base64.urlsafe_b64encode(hashlib.sha256(id_json.encode()).digest()).decode().rstrip("=")
                payload = encode_value(document)
                payload["_mongo_id"] = encode_value(original_id)
                payload["_source_collection"] = source_collection
                target = safe_name(self.collection_prefix + source_collection)
                self.writer.set(self.client.collection(target).document(doc_id), payload)
            except Exception as exc:
                self.parse_fail.inc()
                yield pvalue.TaggedOutput("parse_errors", json.dumps({"error": str(exc), "line": line}, ensure_ascii=False))

        def finish_bundle(self):
            self.writer.close()
            try:
                while True:
                    yield WindowedValue(pvalue.TaggedOutput("write_errors", self.errors.get_nowait()), MAX_TIMESTAMP, (GlobalWindow(),))
            except Empty:
                return

    gcs, bucket, prefix = gcs_parts()
    input_pattern = f"gs://{bucket}/{prefix}/data/*/*.jsonl.gz"
    dead_prefix = env("FIRESTORE_DEAD_LETTER_PREFIX").rstrip("/")
    project = env("GOOGLE_CLOUD_PROJECT")
    opts = [
        "--runner=DataflowRunner", f"--project={project}", f"--region={env('DATAFLOW_REGION')}",
        f"--temp_location={env('DATAFLOW_TEMP_GCS')}", f"--staging_location={env('DATAFLOW_STAGING_GCS')}",
        f"--num_workers={os.getenv('DATAFLOW_NUM_WORKERS', '8')}", f"--max_num_workers={os.getenv('DATAFLOW_MAX_WORKERS', '50')}",
        f"--machine_type={os.getenv('DATAFLOW_MACHINE_TYPE', 'n2-standard-4')}",
        "--autoscaling_algorithm=THROUGHPUT_BASED", "--save_main_session=true",
    ]
    worker_sa = os.getenv("DATAFLOW_WORKER_SERVICE_ACCOUNT")
    if worker_sa:
        opts.append(f"--service_account_email={worker_sa}")
    options = PipelineOptions(opts)
    options.view_as(SetupOptions).requirements_file = str(Path(__file__).with_name("requirements.txt"))
    pipeline = beam.Pipeline(options=options)
    result = (
        pipeline
        | "ReadMongoExport" >> beam.io.ReadFromText(input_pattern, compression_type=CompressionTypes.AUTO)
        | "WriteFirestore" >> beam.ParDo(FirestoreWriter(
            project, env("FIRESTORE_DATABASE", "(default)"),
            int(os.getenv("FIRESTORE_RETRY_LIMIT", "5")),
            os.getenv("FIRESTORE_COLLECTION_PREFIX", "migrated_"),
        )).with_outputs("parse_errors", "write_errors")
    )
    result.parse_errors | "SaveParseErrors" >> beam.io.WriteToText(dead_prefix + "/parse-errors", file_name_suffix=".jsonl")
    result.write_errors | "SaveWriteErrors" >> beam.io.WriteToText(dead_prefix + "/write-errors", file_name_suffix=".jsonl")
    job = pipeline.run()
    final_state = job.wait_until_finish()
    if str(final_state) != "DONE":
        raise RuntimeError(f"Dataflow job ended in {final_state}")
    query = job.metrics().query()
    succeeded = 0
    for metric in query.get("counters", []):
        if metric.key.metric.name == "items_succeeded":
            value = metric.committed if metric.committed is not None else metric.attempted
            succeeded += int(value or 0)
    gcs.bucket(bucket).blob(f"{prefix}/dataflow-result.json").upload_from_string(json.dumps({"job_id": job.job_id(), "state": str(final_state), "items_succeeded": succeeded}, indent=2), content_type="application/json")
    print(json.dumps({"dataflow_job_id": job.job_id(), "state": str(final_state), "items_succeeded": succeeded}))


def validate() -> None:
    client, bucket_name, prefix = gcs_parts()
    bucket = client.bucket(bucket_name)
    manifest_blob = bucket.blob(f"{prefix}/export-manifest.json")
    result_blob = bucket.blob(f"{prefix}/dataflow-result.json")
    if not manifest_blob.exists() or not result_blob.exists():
        raise RuntimeError("Export manifest or Dataflow result is missing")
    expected = sum(json.loads(manifest_blob.download_as_text()).values())
    result = json.loads(result_blob.download_as_text())
    dead_letter_uri = env("FIRESTORE_DEAD_LETTER_PREFIX")
    bucket_prefix = f"gs://{bucket_name}/"
    if not dead_letter_uri.startswith(bucket_prefix):
        raise ValueError("FIRESTORE_DEAD_LETTER_PREFIX must point inside GCS_BUCKET")
    error_blobs = client.list_blobs(bucket_name, prefix=dead_letter_uri.removeprefix(bucket_prefix).rstrip("/") + "/")
    errors = [blob for blob in error_blobs if (blob.size or 0) > 0]
    if errors:
        raise RuntimeError(f"Dead-letter files exist ({len(errors)} objects); inspect gs://{bucket_name}/{errors[0].name}")
    actual = int(result["items_succeeded"])
    if expected != actual:
        raise RuntimeError(f"Validation count mismatch: exported={expected}, Firestore successes={actual}")
    print(json.dumps({"validation": "passed", "exported": expected, "firestore_succeeded": actual}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "ai-review", "export", "load", "validate"))
    stage = parser.parse_args().stage
    {"preflight": preflight, "ai-review": ai_review, "export": export_to_gcs, "load": load_to_firestore, "validate": validate}[stage]()
