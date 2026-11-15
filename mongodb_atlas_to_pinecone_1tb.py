"""Resumable large-scale MongoDB Atlas -> NVIDIA embeddings -> S3 -> Pinecone import.

This is a bulk data pipeline, not a per-record LLM agent loop. It streams Atlas
records, embeds configured text, writes Pinecone's Parquet import format, uploads
shards to S3, checkpoints progress, then starts Pinecone's asynchronous import.
"""
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from bson import json_util
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pymongo import ASCENDING, MongoClient

load_dotenv(Path(__file__).resolve().with_name(".env.mongodb-pinecone-1tb"))

REQUIRED = [
    "MONGODB_ATLAS_URI", "MONGODB_DATABASE", "MONGODB_COLLECTION",
    "PINECONE_API_KEY", "PINECONE_INDEX", "NVIDIA_API_KEY",
    "S3_BUCKET", "S3_IMPORT_PREFIX", "PINECONE_STORAGE_INTEGRATION_ID",
]
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    raise EnvironmentError("Missing .env values: " + ", ".join(missing))

MONGO_URI = os.environ["MONGODB_ATLAS_URI"]
DB_NAME = os.environ["MONGODB_DATABASE"]
COLLECTION_NAME = os.environ["MONGODB_COLLECTION"]
INDEX_NAME = os.environ["PINECONE_INDEX"]
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "mongodb-atlas-1tb-v1")
EMBED_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b")
DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
TEXT_FIELDS = [s.strip() for s in os.getenv("TEXT_FIELDS", "text").split(",") if s.strip()]
METADATA_FIELDS = [s.strip() for s in os.getenv(
    "METADATA_FIELDS", "file_id,filename,page,chunk_index,created_at").split(",") if s.strip()]
MONGO_BATCH_SIZE = int(os.getenv("MONGO_BATCH_SIZE", "500"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
PARQUET_ROWS_PER_FILE = int(os.getenv("PARQUET_ROWS_PER_FILE", "5000"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "5000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))
S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = os.environ["S3_IMPORT_PREFIX"].strip("/")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
STORAGE_INTEGRATION_ID = os.environ["PINECONE_STORAGE_INTEGRATION_ID"]
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("CHECKPOINT_FILE", "mongodb_pinecone_1tb_checkpoint.json"))
WORK_DIR = Path(os.getenv("WORK_DIR", "mongodb_pinecone_parquet_work"))
if not CHECKPOINT.is_absolute():
    CHECKPOINT = SCRIPT_DIR / CHECKPOINT
if not WORK_DIR.is_absolute():
    WORK_DIR = SCRIPT_DIR / WORK_DIR
MAX_IMPORT_GB = float(os.getenv("MAX_IMPORT_GB", "900"))  # Safety buffer below Pinecone on-demand 1 TB limit.

if CHUNK_CHARS < 200 or not 0 <= CHUNK_OVERLAP < CHUNK_CHARS:
    raise ValueError("Require CHUNK_CHARS >= 200 and 0 <= CHUNK_OVERLAP < CHUNK_CHARS")
if min(MONGO_BATCH_SIZE, EMBED_BATCH_SIZE, PARQUET_ROWS_PER_FILE) < 1:
    raise ValueError("Batch sizes must be positive")

embedder = NVIDIAEmbeddings(
    model=EMBED_MODEL,
    api_key=os.environ["NVIDIA_API_KEY"],
    dimensions=DIMENSION,
    trucate="END",
)


def get_path(obj: Any, path: str):
    value = obj
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


def document_text(doc: dict) -> str:
    return "\n".join(
        f"{field}: {stringify(value)}"
        for field in TEXT_FIELDS
        if (value := get_path(doc, field)) is not None and stringify(value).strip()
    ).strip()


def split_text(text: str):
    text = text.strip()
    if not text:
        return []
    pieces, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + CHUNK_CHARS // 2, end)
            if boundary > start:
                end = boundary
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return [p for p in pieces if p]


def stable_vector_id(source_id: str, part: int) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL,
                      f"{DB_NAME}.{COLLECTION_NAME}:{source_id}:{part}").hex


def metadata_for(doc: dict, source_id: str, chunk_no: int, text: str) -> str:
    metadata = {
        "source_database": DB_NAME,
        "source_collection": COLLECTION_NAME,
        "mongo_id": source_id,
        "chunk_index": chunk_no,
        "text": text,
    }
    for field in METADATA_FIELDS:
        val = get_path(doc, field)
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            metadata[field.replace(".", "_")] = val
        elif isinstance(val, (datetime, date)):
            metadata[field.replace(".", "_")] = val.isoformat()
        elif isinstance(val, list) and all(isinstance(x, (str, int, float, bool)) for x in val):
            metadata[field.replace(".", "_")] = val
        else:
            metadata[field.replace(".", "_")] = stringify(val)
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))


def load_checkpoint():
    if not CHECKPOINT.exists():
        return {"last_id": None, "next_file": 0, "docs_processed": 0,
                "vectors_staged": 0, "skipped_empty": 0, "staged_bytes": 0,
                "stage_complete": False, "import_id": None}
    return json.loads(CHECKPOINT.read_text(encoding="utf-8"))


def save_checkpoint(state):
    tmp = CHECKPOINT.with_suffix(CHECKPOINT.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, CHECKPOINT)


def mongo_last_id(state):
    raw = state.get("last_id")
    return json_util.loads(raw) if raw else None


def save_source_id(state, source_id):
    state["last_id"] = json_util.dumps(source_id)


def make_pinecone():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(INDEX_NAME):
        print(f"Creating dense AWS Pinecone index {INDEX_NAME} ({DIMENSION} dimensions)")
        pc.create_index(
            name=INDEX_NAME,
            vector_type="dense",
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
    desc = pc.describe_index(INDEX_NAME)
    dim = getattr(desc, "dimension", None)
    if dim is None and isinstance(desc, dict):
        dim = desc.get("dimension")
    if int(dim) != DIMENSION:
        raise ValueError(f"Index dimension {dim} does not match embedding dimension {DIMENSION}")
    return pc, pc.Index(INDEX_NAME)


def retry_call(fn, label):
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(x in msg for x in (
                "429", "500", "502", "503", "504", "timeout", "temporarily",
                "connection reset", "service unavailable", "throttl",
            ))
            if not transient or attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"{label} failed after {attempt + 1} attempt(s): {exc}") from exc
            delay = min(60, 2 ** attempt) + (attempt % 3)
            print(f"{label} transient failure; retry {attempt + 1}/{MAX_RETRIES} in {delay}s")
            time.sleep(delay)


def s3_client():
    return boto3.client("s3", region_name=S3_REGION)


def object_prefix():
    return f"{S3_PREFIX}/{NAMESPACE}/" if S3_PREFIX else f"{NAMESPACE}/"


def clean_uncommitted_s3_shards(s3, state):
    """Remove only uncheckpointed shard objects, so interrupted doc writes can resume safely."""
    prefix = object_prefix()
    paginator = s3.get_paginator("list_objects_v2")
    stale = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            match = re.search(r"part-(\d{9})\.parquet$", item["Key"])
            if match and int(match.group(1)) >= int(state["next_file"]):
                stale.append({"Key": item["Key"]})
    if stale and not CHECKPOINT.exists():
        raise RuntimeError(
            f"Found staged Parquet shards at s3://{S3_BUCKET}/{prefix} but no checkpoint file. "
            "Use a new S3 prefix or restore the matching checkpoint; refusing to delete staged data."
        )
    for start in range(0, len(stale), 1000):
        s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": stale[start:start + 1000]})


def write_parquet_shard(rows, state, s3):
    if not rows:
        return
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    file_no = int(state["next_file"])
    local_file = WORK_DIR / f"part-{file_no:09d}.parquet"
    schema = pa.schema([
        ("id", pa.string()),
        ("values", pa.list_(pa.float32())),
        ("metadata", pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, local_file, compression="zstd", row_group_size=min(1000, len(rows)))
    key = object_prefix() + local_file.name
    retry_call(lambda: s3.upload_file(str(local_file), S3_BUCKET, key), f"S3 upload {local_file.name}")
    size = local_file.stat().st_size
    state["next_file"] = file_no + 1
    state["vectors_staged"] += len(rows)
    state["staged_bytes"] += size
    local_file.unlink(missing_ok=True)
    print(f"Staged file {file_no:09d}: {len(rows):,} vectors, {size / (1024**2):.1f} MiB")


def stage_collection(state, collection, s3):
    clean_uncommitted_s3_shards(s3, state)
    last_id = mongo_last_id(state)
    query = {"_id": {"$gt": last_id}} if last_id is not None else {}
    cursor = collection.find(query, no_cursor_timeout=False).sort("_id", ASCENDING).batch_size(MONGO_BATCH_SIZE)
    rows = []
    pending_docs = 0
    pending_empty = 0
    pending_last_id = None
    pool = ThreadPoolExecutor(max_workers=EMBED_WORKERS)

    def embed_one_batch(text_batch):
        vectors = retry_call(lambda: embedder.embed_documents(text_batch), "NVIDIA embedding request")
        if len(vectors) != len(text_batch):
            raise RuntimeError("Embedding response size did not match input batch")
        return vectors

    try:
        for doc in cursor:
            source_id = str(doc.get("_id"))
            pieces = split_text(document_text(doc))
            if not pieces:
                pending_empty += 1
            else:
                batches = [(first, pieces[first:first + EMBED_BATCH_SIZE])
                           for first in range(0, len(pieces), EMBED_BATCH_SIZE)]
                vector_batches = pool.map(lambda batch: embed_one_batch(batch[1]), batches)
                for (first, text_batch), vectors in zip(batches, vector_batches):
                    for offset, (piece, vector) in enumerate(zip(text_batch, vectors)):
                        if len(vector) != DIMENSION:
                            raise RuntimeError(f"Embedding dimension {len(vector)} != configured {DIMENSION}")
                        chunk_no = first + offset
                        rows.append({
                            "id": stable_vector_id(source_id, chunk_no),
                            "values": vector,
                            "metadata": metadata_for(doc, source_id, chunk_no, piece),
                        })
            pending_docs += 1
            pending_last_id = doc["_id"]
            # Only checkpoint after all vectors for a whole document are safely in
            # S3. Otherwise, an interruption could skip in-memory rows on resume.
            if len(rows) >= PARQUET_ROWS_PER_FILE:
                write_parquet_shard(rows, state, s3)
                rows = []
                state["docs_processed"] += pending_docs
                state["skipped_empty"] += pending_empty
                save_source_id(state, pending_last_id)
                save_checkpoint(state)
                pending_docs = pending_empty = 0
                pending_last_id = None
                print(f"Progress: {state['docs_processed']:,} source docs; "
                      f"{state['vectors_staged']:,} vectors staged")
            elif pending_docs >= 1000:
                # Bound restart work during long stretches of empty documents.
                # If there are pending vectors, persist them before checkpointing.
                if rows:
                    write_parquet_shard(rows, state, s3)
                    rows = []
                state["docs_processed"] += pending_docs
                state["skipped_empty"] += pending_empty
                save_source_id(state, pending_last_id)
                save_checkpoint(state)
                pending_docs = pending_empty = 0
                pending_last_id = None
        if rows:
            write_parquet_shard(rows, state, s3)
            rows = []
        if pending_docs:
            state["docs_processed"] += pending_docs
            state["skipped_empty"] += pending_empty
            save_source_id(state, pending_last_id)
            save_checkpoint(state)
        state["stage_complete"] = True
        save_checkpoint(state)
    finally:
        cursor.close()
        pool.shutdown(wait=True)


def s3_staged_size(s3):
    total = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=object_prefix()):
        total += sum(obj["Size"] for obj in page.get("Contents", []))
    return total


def start_or_resume_import(index, state):
    if not state.get("import_id"):
        stats = index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", None)
        if namespaces is None and isinstance(stats, dict):
            namespaces = stats.get("namespaces", {})
        if NAMESPACE in (namespaces or {}):
            raise RuntimeError(
                f"Namespace '{NAMESPACE}' already exists. Pinecone bulk import requires a new namespace. "
                "Set a fresh PINECONE_NAMESPACE and restage to a new S3 namespace folder."
            )
        uri = f"s3://{S3_BUCKET}/{S3_PREFIX}" if S3_PREFIX else f"s3://{S3_BUCKET}"
        result = retry_call(lambda: index.start_import(
            uri=uri,
            integration_id=STORAGE_INTEGRATION_ID,
            error_mode="abort",
        ), "Pinecone start_import")
        state["import_id"] = str(result.id)
        save_checkpoint(state)
        print(f"Pinecone import started: {state['import_id']}")

    import_id = state["import_id"]
    deadline = time.monotonic() + float(os.getenv("IMPORT_TIMEOUT_HOURS", "72")) * 3600
    while True:
        status = retry_call(lambda: index.describe_import(import_id), "Pinecone import status")
        state_name = str(status.status)
        imported = int(status.records_imported or 0)
        print(f"Import {import_id}: {state_name}; {status.percent_complete or 0:.1f}% "
              f"({imported:,} records)")
        if state_name == "Completed":
            if imported != int(state["vectors_staged"]):
                raise RuntimeError(
                    f"Pinecone imported {imported:,}, but staged {state['vectors_staged']:,} vectors. "
                    "Inspect the import and source before retrying."
                )
            state["import_complete"] = True
            state["records_imported"] = imported
            save_checkpoint(state)
            return
        if state_name in ("Failed", "Cancelled"):
            raise RuntimeError(f"Pinecone import {state_name}: {getattr(status, 'error', None)}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Pinecone import {import_id} is still {state_name}; rerun to resume monitoring")
        time.sleep(30)


def main():
    state = load_checkpoint()
    pc, index = make_pinecone()
    s3 = s3_client()
    if not state["stage_complete"]:
        mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000)
        try:
            mongo.admin.command("ping")
            collection = mongo[DB_NAME][COLLECTION_NAME]
            source_count = collection.count_documents({})
            if source_count == 0:
                raise RuntimeError(f"Source collection {DB_NAME}.{COLLECTION_NAME} is empty")
            print(f"Source: {DB_NAME}.{COLLECTION_NAME}; {source_count:,} docs; "
                  f"resume point: {state['docs_processed']:,} docs")
            stage_collection(state, collection, s3)
        finally:
            mongo.close()

    if int(state["vectors_staged"]) == 0:
        raise RuntimeError(
            "No vectors were staged. Check that TEXT_FIELDS exists in the source documents before importing."
        )
    staged_bytes = s3_staged_size(s3)
    staged_gb = staged_bytes / (1000**3)
    print(f"Parquet staged: {state['vectors_staged']:,} vectors; {staged_gb:,.2f} GB")
    if MAX_IMPORT_GB > 0 and staged_gb > MAX_IMPORT_GB:
        raise RuntimeError(
            f"Staged import is {staged_gb:.2f} GB, above MAX_IMPORT_GB={MAX_IMPORT_GB}. "
            "Keep this staged data, review Pinecone import limits/plan, then raise the configured limit only if supported."
        )
    start_or_resume_import(index, state)
    print("Migration completed.")
    print(json.dumps({
        "source_collection": f"{DB_NAME}.{COLLECTION_NAME}",
        "source_docs_processed": state["docs_processed"],
        "empty_docs_skipped": state["skipped_empty"],
        "vectors_staged": state["vectors_staged"],
        "vectors_imported": state.get("records_imported"),
        "index": INDEX_NAME,
        "namespace": NAMESPACE,
        "embedding_model": EMBED_MODEL,
        "embedding_dimension": DIMENSION,
        "s3_import_uri": f"s3://{S3_BUCKET}/{S3_PREFIX}",
    }, indent=2))


if __name__ == "__main__":
    main()
