# Aurora MySQL to Cloud Spanner with Dataflow

This is an offline, one-time migration starter based on the supplied script. It inventories Aurora MySQL, asks `nvidia/nemotron-3-ultra-550b-a55b` to assess migration risks, creates a basic GoogleSQL schema, launches Google's **SourceDB to Spanner** Dataflow batch template, waits for the job to finish, and compares exact row counts.

## Configure

1. Create a Spanner instance and a **GoogleSQL-dialect** database. The script does not provision Google Cloud resources.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Authenticate on the machine running this script with Application Default Credentials. The identity needs permission to launch Dataflow jobs, read the source connection settings, and query Spanner. The Dataflow worker service account needs access to Spanner, the staging/temp/output buckets, and the Aurora network.
4. Ensure Dataflow workers can reach the Aurora endpoint. Configure VPC/subnetwork and routing as needed; set `DATAFLOW_SUBNETWORK` where required. Aurora security groups must allow worker traffic on port 3306.
5. Set these environment variables (for example, in a local `.env` file):

```dotenv
MYSQL_HOST=your-aurora-writer-endpoint
MYSQL_PORT=3306
MYSQL_USER=migration_user
MYSQL_PASSWORD=replace-me
MYSQL_DATABASE=source_db
NVIDIA_API_KEY=your-nvidia-api-key
GOOGLE_CLOUD_PROJECT=your-project-id
SPANNER_INSTANCE=your-instance-id
SPANNER_DATABASE=your-database-id
DATAFLOW_REGION=us-central1
DATAFLOW_OUTPUT_GCS=gs://your-bucket/migration-errors
DATAFLOW_TEMP_GCS=gs://your-bucket/dataflow-temp
DATAFLOW_STAGING_GCS=gs://your-bucket/dataflow-staging
DATAFLOW_MAX_WORKERS=10
DATAFLOW_TEMPLATE_VERSION=latest
# Optional:
# DATAFLOW_SUBNETWORK=regions/us-central1/subnetworks/your-subnet
# DATAFLOW_SERVICE_ACCOUNT=dataflow-worker@your-project-id.iam.gserviceaccount.com
```

6. Review the generated Spanner schema and type mappings before migration. Dataflow requires the compatible Spanner tables to exist first.
7. Run: `python aurora_mysql_to_spanner.py`

The script submits the Dataflow REST launch request, waits for a terminal job state, then compares table counts. Credentials are sent to the Dataflow API in the launch request; protect the host running the script and its environment. For stricter secret handling, use the template's Cloud KMS encrypted credential parameters.

## Review before production

- Google's **SourceDB to Spanner** template is currently Beta/Pre-GA and supports only a limited set of MySQL types. Confirm your schema/data are supported and inspect failed/skipped records in `DATAFLOW_OUTPUT_GCS`.
- Schema conversion in this script is intentionally basic; it is not the Spanner Migration Tool. Spanner requires primary keys. Tables without one stop schema generation.
- MySQL `AUTO_INCREMENT` is not recreated. Existing source key values are copied, but new application writes need a Spanner-compatible key strategy. Monotonically increasing keys may create write hotspots.
- MySQL column defaults, generated columns, `ON UPDATE`, collations, partitioning, checks, and engine options are not translated. Review `SHOW CREATE TABLE` for every table.
- `TIME` becomes `STRING(32)`, `ENUM`/`SET` become strings, and oversized decimal types become strings. Verify the `TINYINT(1)` to `BOOL` mapping. Foreign keys and indexes may need adjustment for Spanner.
- Stored procedures, functions, and triggers need application-side rewrites. Views and query syntax need separate conversion.
- This is a batch copy, not CDC. For a live source, writes during the job can cause count mismatches or inconsistent copied data. Plan a write pause/cutover or a separate CDC migration.
- Row-count equality is a basic check, not proof that values or application behavior match. Validate critical data at the row/content level before cutover.
- Dataflow template `latest` can change. Pin a dated template version for a production run after validating it.


