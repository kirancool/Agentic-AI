# AWS DynamoDB to Cloud Firestore: sequential AI-agent migration

This kit runs the migration automatically, one AI agent after another, from a single Python command. Each agent has only its stage-specific tool; the Python workflow enforces the order and checks each result before it starts the next stage. The orchestrator uses `nvidia/nemotron-3-ultra-550b-a55b`.

The agents run these stages in order:

1. **Preflight agent** checks that the table is active, PITR is enabled, and required configuration exists. Nemotron reviews key schema, indexes, and mapping risks.
2. **Export agent** starts a DynamoDB point-in-time export to S3 and waits for completion.
3. **Transfer agent** starts a managed Storage Transfer Service copy from S3 to GCS and polls until it completes.
4. **Dataflow agent** launches the Python Apache Beam pipeline, monitors the job, and retrieves success/failure metrics.
5. **Validation agent** compares the AWS export item count with successful Firestore writes and checks both dead-letter locations.

The Dataflow loader is Python only and writes through Firestore's Python `BulkWriter`. No Java or Maven is required. Google documents Firestore's Python bulk writer for large migrations and recommends Dataflow for large-scale processing. [BulkWriter reference](https://docs.cloud.google.com/python/docs/reference/firestore/latest/google.cloud.firestore_v1.bulk_writer.BulkWriter) · [Firestore Dataflow guidance](https://docs.cloud.google.com/firestore/native/docs/dataflow-connector)

## Configure

1. DynamoDB point-in-time recovery (PITR) must be enabled. The AWS identity needs `DescribeTable`, `DescribeContinuousBackups`, `ExportTableToPointInTime`, and `DescribeExport`; DynamoDB also needs permission to write the export to S3.
2. Create a read-only AWS credentials JSON for Storage Transfer Service, based on `aws-transfer-creds.example.json`. For S3 objects encrypted with SSE-KMS, allow the transfer identity to decrypt with that key.
3. Create the Firestore database and GCS bucket(s). Enable Dataflow, Firestore, Cloud Storage, and Storage Transfer APIs. The Dataflow worker service account needs Firestore write access and GCS read/write access.
4. Authenticate the local submitter with AWS credentials for DynamoDB and Google Application Default Credentials for Google APIs.
5. Copy `.env.example` to `.env`, copy the credentials example to `aws-transfer-creds.json`, and set the values. Do not commit either secret file.

## Start the full process

From this directory in PowerShell:

```powershell
py -m pip install -r requirements-agentic.txt
Copy-Item .env.example .env
Copy-Item aws-transfer-creds.example.json aws-transfer-creds.json
# Edit .env and aws-transfer-creds.json, then run:
py agentic_migration.py
```

The orchestrator waits for each stage, saves progress to `migration_state.json`, and stops if a stage fails. Run `py agentic_migration.py` again to resume from the last incomplete stage. Use `py agentic_migration.py --new-run` to start a new point-in-time export and migration.

The process writes a per-run export/validation report to `migration_<run-id>_export.json`. The Dataflow worker writes malformed rows and permanent Firestore write failures to separate GCS dead-letter prefixes. The pipeline retries failed writes up to `FIRESTORE_RETRY_LIMIT` and measures successful writes.

## Large dataset notes (~1 TB)

- Worker counts, machine type, Firestore rate limits, and transfer timeout in `.env.example` are starting values. Pilot a representative sample and tune using observed Dataflow throughput, Firestore latency, and quotas; they do not guarantee a completion time.
- DynamoDB export emits multiple compressed files, which Beam reads in parallel. Do not combine them into one large gzip object.
- Firestore documents are limited to 1 MiB, field names to 1,500 bytes, and map/array nesting depth to 20. Failures are written to dead-letter paths. See [Firestore quotas](https://docs.cloud.google.com/firestore/quotas).
- The validation agent requires the source export item count and Dataflow `items_succeeded` counter to match and requires both dead-letter prefixes to be empty. If metrics are unavailable or counts differ, the workflow stops instead of reporting success.
- This is a point-in-time snapshot migration. Writes after the export time are not captured. For live cutover, pause source writes or add a separate DynamoDB Streams/incremental-export change-capture and reconciliation stage.

## Type and data-model mapping

- DynamoDB `S`, `BOOL`, `NULL`, `B`, `M`, and `L` map to Firestore strings, booleans, nulls, bytes, maps, and arrays. Sets become arrays; Firestore does not enforce set uniqueness.
- Nested lists are wrapped as `{ "ddb_type": "L", "items": [...] }` maps when they occur inside another list, because Firestore does not allow arrays directly inside arrays.
- Numbers that cannot be represented exactly as Firestore int64/double are kept as strings to avoid precision loss.
- Attributes with Firestore-reserved names matching `__.*__` are sent to the parse dead-letter output. Update the converter if your table uses these names.
- DynamoDB GSIs, query patterns, TTL, and access rules do not automatically become Firestore indexes, query patterns, TTL policies, or security rules. Design those for the destination application.
- Writes are deterministic upserts using IDs derived from partition/sort keys. Reruns overwrite matching documents and do not delete destination-only documents.
