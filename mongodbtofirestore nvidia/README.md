# Amazon DocumentDB to Cloud Firestore (Python + GitHub Actions)

This kit uses a manually triggered GitHub Actions workflow to run five sequential NVIDIA Nemotron specialist agents through NVIDIA's OpenAI-compatible Chat Completions API: preflight, schema review, DocumentDB export, Dataflow load, and validation. Each agent gets one stage-specific Python function tool. The controller requires that tool call, runs the deterministic Python stage, and starts the next agent only after success. Parallel stage calls are disabled. The model ID is `nvidia/nemotron-3-ultra-550b-a55b`; the NVIDIA endpoint is `https://integrate.api.nvidia.com/v1`. No Java is used. The workflow is deliberately manual because it writes to the destination; starting it requires typing `MIGRATE`.

The schema-review agent receives collection names, estimated document counts, and index metadata only. It does not receive document values or modify records. Its report is saved to `GCS_PREFIX/ai-review.json`. Agents coordinate and report the stages; the Python controller enforces order, stops on errors, and handles all data operations.

## Architecture

```text
GitHub Actions (OIDC to AWS and GCP)
  -> self-hosted Linux runner in the DocumentDB VPC
  -> PyMongo reads DocumentDB in batches and uploads bounded gzip chunks to GCS
  -> Dataflow Python workers read those chunks and upsert Firestore documents
  -> sequential Nemotron stage agents call Python functions; Python checks counts and dead-letter output
```

The runner must resolve and connect to the private DocumentDB cluster, and reach GitHub, AWS STS, GCS, and Google APIs. It needs enough local free disk for one compressed chunk (default 128 MiB, plus working space), not for the full export. Dataflow workers need access to the GCS paths and Firestore write permission. The workflow uses GitHub OIDC to assume an AWS IAM role; PyMongo's `MONGODB-AWS` mechanism authenticates to DocumentDB using the resulting temporary credentials. No static AWS access keys or DocumentDB password are stored. This requires an instance-based Amazon DocumentDB 5.0 cluster with IAM authentication enabled. See [AWS DocumentDB IAM authentication](https://docs.aws.amazon.com/documentdb/latest/devguide/iam-identity-auth.html).

## 1. Prepare AWS and Google Cloud

1. Create or select a Linux x64 GitHub self-hosted runner in the AWS VPC/subnets that can reach the DocumentDB cluster. Apply the custom runner label `docdb-vpc`. Configure security groups and routes so it can reach the DocumentDB endpoint on port 27017 and Google/GitHub endpoints over HTTPS.
2. Enable IAM authentication for an instance-based DocumentDB 5.0 cluster and create a database principal in `$external` for the GitHub-assumed role ARN, granting only read access to the source database. AWS's example is `db.createUser({user: "arn:aws:iam::ACCOUNT:role/ROLE", mechanisms: ["MONGODB-AWS"], roles: [{role: "read", db: "YOUR_DATABASE"}]})`. The workflow downloads AWS's public DocumentDB CA bundle into `/tmp/global-bundle.pem` and configures TLS, `replicaSet=rs0`, `readPreference=secondaryPreferred`, `retryWrites=false`, and `authMechanism=MONGODB-AWS`.
3. Create a GCS bucket, Firestore database, and Dataflow worker service account. Enable the Cloud Storage, Firestore, Dataflow, and IAM Credentials APIs. Grant the GitHub migration service account permissions to submit/monitor Dataflow jobs and access the bucket. Grant the Dataflow worker service account GCS read/write and Firestore write access.
4. Configure Google Workload Identity Federation to trust this GitHub repository and allow the federated identity to impersonate the migration service account. Use Google's [GitHub Actions authentication guide](https://github.com/google-github-actions/auth#workload-identity-federation).

## 2. Add GitHub repository configuration

Add this **Actions secret**:

- `NVIDIA_API_KEY`: NVIDIA API key with access to the Nemotron endpoint

Add these **Actions variables** (Settings -> Secrets and variables -> Actions -> Variables):

- `DOCDB_DATABASE`
- `DOCDB_HOST`, `AWS_REGION`, and `AWS_ROLE_ARN` (the GitHub OIDC role ARN; it must match the `$external` DocumentDB principal)
- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `GCP_WORKLOAD_IDENTITY_PROVIDER` (full provider resource name)
- `GCP_MIGRATION_SERVICE_ACCOUNT` (service account email to impersonate)
- `DATAFLOW_REGION`
- `DATAFLOW_TEMP_GCS` and `DATAFLOW_STAGING_GCS` (full `gs://...` paths)
- `DATAFLOW_WORKER_SERVICE_ACCOUNT`

Optional variables: `FIRESTORE_DATABASE` (defaults to `(default)`), `DATAFLOW_NUM_WORKERS`, `DATAFLOW_MAX_WORKERS`, `DATAFLOW_MACHINE_TYPE`, `FIRESTORE_RETRY_LIMIT`, `FIRESTORE_COLLECTION_PREFIX`, `MONGO_EXPORT_CHUNK_MB`, `MONGO_EXPORT_BATCH_SIZE`. The example values in the workflow are starting points; tune against observed throughput and quotas.

The GitHub-assumed role ARN must match the principal configured in the DocumentDB `$external` database. Configure its AWS trust policy for GitHub's OIDC provider and restrict it to this repository and branch/environment. Local runs can use temporary AWS IAM Identity Center credentials or an attached role; PyMongo requires the `aws` extra (`pymongo[aws]`) for `MONGODB-AWS` authentication.

Copy `.github/workflows/migrate.yml` into the repository's `.github/workflows/` directory, and copy this package's Python files and `requirements.txt` into `outputs/mongodb_to_firestore/` (or adjust the workflow paths). The workflow YAML is already in that directory for convenient review.

## 3. Run it

1. Push the files to the repository's default branch.
2. In GitHub, open **Actions -> Amazon DocumentDB to Firestore -> Run workflow**.
3. Type `MIGRATE` and start the workflow.
4. Follow each stage in the Actions log. Review `GCS_PREFIX/ai-review.json` before cutover. On success, check the export and Dataflow summary JSON objects under `GCS_PREFIX` and confirm Firestore document counts.

The run-specific GCS prefix is `mongo-firestore/<github-run-id>-<attempt>`. A rerun creates a new prefix and re-exports the source. It does not resume a partially completed export. Destination writes are deterministic upserts; reruns overwrite matching IDs but do not remove destination-only documents.

## Scale and consistency limits

- GitHub-hosted runners cannot reach a private DocumentDB cluster and GitHub-hosted jobs have a six-hour execution limit. The included workflow therefore targets a self-hosted runner in the VPC, but the configured job is still capped at six hours. A 1 TB migration may take longer depending on document count, indexes, cluster load, network bandwidth, and Firestore quotas. For a run expected to exceed six hours, use GitHub Actions to submit a long-running AWS Batch/ECS task or a remote migration controller and have that service execute/resume the Python export stages; do not rely on a single Actions job.
- This is a collection-by-collection scan, not a point-in-time database snapshot. Pause source writes during export/cutover or add a separate change-stream capture and reconciliation step. Validate consistency before switching applications.
- MongoDB indexes, validators, TTL behavior, aggregation pipelines, and query semantics are not migrated. Plan Firestore collections, indexes, security rules, and application query changes separately.
- The loader converts common BSON values: ObjectId to string, Decimal128 to string, binary to bytes, and BSON dates to timestamps. Nested arrays are wrapped because Firestore does not support arrays directly inside arrays. Unsupported or invalid documents are written to a dead-letter output.
- Firestore has a 1 MiB document size limit and field/map depth limits. Review the [Firestore quota and limits page](https://docs.cloud.google.com/firestore/quotas). The pipeline stops validation if write counts differ or any dead-letter files exist.
- Dataflow handles parallel ingestion; the initial source export is bounded-memory but sequential per collection. For very large collections, source-export throughput may be the bottleneck. Tune chunk and cursor batch sizes only after measuring DocumentDB load.

## Local commands

From this package directory, configure the same environment variables and Google ADC, install dependencies, then run stages manually in order:

```bash
python -m pip install -r requirements.txt
python migration.py preflight
python migration.py ai-review
python migration.py export
python migration.py load
python migration.py validate
```

The GitHub Actions workflow runs `python agentic_migration.py`; that script runs the five Nemotron agents sequentially and stops at the first failed Python stage. Store the key as a GitHub Actions secret; the workflow injects it into `NVIDIA_API_KEY`. Never commit the key or put it in a workflow variable. For local use, provide the key through your shell or an untracked `.env` file.
