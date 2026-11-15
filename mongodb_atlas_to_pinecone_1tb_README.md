# MongoDB Atlas → Pinecone large migration

## What this pipeline does

The script streams MongoDB documents in `_id` order, reads the configured text fields, splits long text into overlapping chunks, generates NVIDIA embeddings, writes Pinecone-compatible Parquet shards, and uploads those shards to S3. It checkpoints completed source documents and resumes from the last committed `_id` after interruptions. Once staging completes, it starts and monitors Pinecone's asynchronous bulk import.

The migration stores vectors plus selected metadata and chunk text in Pinecone. It does not copy arbitrary MongoDB BSON documents as-is. For the existing `document_chunks` collection, the template uses `TEXT_FIELDS=text` and preserves `file_id`, `filename`, `page`, `chunk_index`, and `created_at` as metadata.

For a 1 TB job, the embedding loop is deterministic and bounded; Nemotron is used as the embedding model, not asked to inspect every row. This keeps the data path resumable and avoids an LLM deciding what to do per record.

## Before the full migration

The earlier inventory found only **289 documents** in `kiran.document_chunks`, not 1 TB. Verify that the Atlas URI, database, and collection point to the intended large dataset. Run a small pilot on representative data first, then measure its document count, vector count, staged Parquet size, and throughput before committing to the full run.

Pinecone recommends object-storage import for 10 million or more records. Its import flow uses Parquet for legacy vector indexes, requires a Pinecone storage integration for a private bucket, and imports into a namespace that does not already exist. Bulk import is available on Standard and Enterprise plans. The script defaults to an AWS-hosted dense serverless index, a fresh namespace, and an import-size guard of 900 GB. On-demand imports have a 1 TB per-import input limit; larger jobs need a supported Dedicated Read Nodes configuration or guidance from Pinecone support. See [Pinecone ingestion overview](https://docs.pinecone.io/guides/data/data-ingestion-overview), [import requirements](https://docs.pinecone.io/guides/data/import-data), and [vector Parquet format](https://docs.pinecone.io/guides/legacy/vectors-api/import-vectors).

The size of the resulting vectors and Parquet files can be substantially larger than the source text. Do not assume that a 1 TB Atlas database will fit in one Pinecone import. The script stages data first and refuses to start import when the configured guard is exceeded; it leaves staged files intact for planning.

## Setup

1. Create an AWS S3 bucket and a Pinecone storage integration that grants Pinecone read access to that bucket. Use an AWS-hosted Pinecone index in a compatible region.
2. Fill `.env.mongodb-pinecone-1tb` with Atlas, NVIDIA, Pinecone, S3, and AWS settings. Prefer an AWS IAM role or profile over long-lived access keys where available.
3. Use a **new, empty Pinecone namespace** for this import. Pinecone bulk import does not load into an existing namespace.
4. Install or upgrade the packages in the same Python environment used to run the script:

   ```bash
   pip install --upgrade pymongo pinecone langchain-nvidia-ai-endpoints python-dotenv boto3 pyarrow
   ```

5. Run:

   ```bash
   python mongodb_atlas_to_pinecone_1tb.py
   ```

The checkpoint JSON and local Parquet work folder are stored beside the script by default. Preserve the checkpoint between restarts. Do not delete or edit the checkpoint or S3 staging prefix during a run. To intentionally start from zero, use a new checkpoint file, S3 prefix, and Pinecone namespace.

## Operational notes

- Source writes during the long scan are not captured. Pause source writes or implement a separate change-stream catch-up phase before cutover.
- Keep `TEXT_FIELDS` and the embedding model/dimension consistent with the application's query embedding code. The template uses NVIDIA `nvidia/nemotron-3.5-embed-1b` at 1024 dimensions to fit the existing 1024-dimensional Pinecone index.
- A 1 TB migration can take a long time and consume substantial NVIDIA embedding, S3 storage/transfer, and Pinecone import capacity. Check account quotas and costs before running the full dataset.
- The script was prepared from your configuration and the prior run logs; it was not run against the 1 TB source. Validate the pilot output before the production-sized run.
