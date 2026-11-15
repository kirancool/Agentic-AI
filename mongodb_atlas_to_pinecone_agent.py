"""Agentic MongoDB Atlas -> Pinecone vector migration.

The LLM agent inspects the MongoDB collection, orchestrates migration tools, and
summarizes exceptions. Documents are chunked, embedded with NVIDIA embeddings,
and deterministically upserted to Pinecone so reruns overwrite the same vectors.
"""
import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pinecone import Pinecone, ServerlessSpec
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()

REQUIRED = ["MONGODB_ATLAS_URI", "MONGODB_DATABASE", "MONGODB_COLLECTION",
            "PINECONE_API_KEY", "PINECONE_INDEX", "NVIDIA_API_KEY"]
missing = [key for key in REQUIRED if not os.getenv(key)]
if missing:
    raise EnvironmentError("Missing .env values: " + ", ".join(missing))

MONGODB_URI = os.environ["MONGODB_ATLAS_URI"]
DB_NAME = os.environ["MONGODB_DATABASE"]
COLLECTION_NAME = os.environ["MONGODB_COLLECTION"]
INDEX_NAME = os.environ["PINECONE_INDEX"]
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "mongodb-atlas-migration")
EMBED_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b")
EMBED_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TEXT_FIELDS = [x.strip() for x in os.getenv("TEXT_FIELDS", "").split(",") if x.strip()]
METADATA_FIELDS = [x.strip() for x in os.getenv("METADATA_FIELDS", "").split(",") if x.strip()]

if CHUNK_CHARS < 200 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_CHARS:
    raise ValueError("Set CHUNK_CHARS >= 200 and 0 <= CHUNK_OVERLAP < CHUNK_CHARS")

chat = ChatNVIDIA(
    model=os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=0,
    max_completion_tokens=4096,
    timeout=300.0,
)
embeddings = NVIDIAEmbeddings(
    model=EMBED_MODEL,
    api_key=os.environ["NVIDIA_API_KEY"],
    dimensions=EMBED_DIMENSION,
)


def mongo_collection():
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=20000)
    client.admin.command("ping")
    return client, client[DB_NAME][COLLECTION_NAME]


def pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(INDEX_NAME):
        print(f"Pinecone index '{INDEX_NAME}' does not exist; creating it...")
        pc.create_index(
            name=INDEX_NAME,
            vector_type="dense",
            dimension=EMBED_DIMENSION,
            metric=os.getenv("PINECONE_METRIC", "cosine"),
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
    description = pc.describe_index(INDEX_NAME)
    dimension = getattr(description, "dimension", None)
    if dimension is None and isinstance(description, dict):
        dimension = description.get("dimension")
    if dimension and int(dimension) != EMBED_DIMENSION:
        raise ValueError(
            f"Pinecone index dimension is {dimension}, but {EMBED_MODEL} is configured "
            f"as {EMBED_DIMENSION}. Create/use an index with matching dimension."
        )
    return pc.Index(INDEX_NAME)


def get_path(document: dict, path: str):
    value: Any = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


def select_document_text(doc: dict) -> str:
    fields = TEXT_FIELDS
    if fields:
        parts = [f"{field}: {stringify(get_path(doc, field))}" for field in fields
                 if get_path(doc, field) is not None]
    else:
        # Fallback for unknown schemas: embed all non-ID scalar/object fields.
        parts = [f"{key}: {stringify(value)}" for key, value in doc.items()
                 if key != "_id" and value is not None]
    return "\n".join(parts).strip()


def clean_metadata_value(value: Any):
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        cleaned = [clean_metadata_value(x) for x in value]
        return [x for x in cleaned if isinstance(x, (str, bool, int, float))]
    # Pinecone metadata values cannot be nested objects; serialize them as JSON text.
    return stringify(value)


def chunks(text: str):
    text = text.strip()
    if not text:
        return []
    out = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return out


def vector_id(mongo_id: str, chunk_no: int) -> str:
    raw = f"{DB_NAME}.{COLLECTION_NAME}:{mongo_id}:{chunk_no}"
    return uuid.uuid5(uuid.NAMESPACE_URL, raw).hex


@tool
def inspect_mongodb_collection(sample_size: int = 5) -> str:
    """Inspect collection count, sample document field names, and candidate text fields."""
    client, collection = mongo_collection()
    try:
        total = collection.count_documents({})
        sample = list(collection.find({}, limit=max(1, min(sample_size, 10))))
        field_names = sorted({key for doc in sample for key in doc.keys()})
        examples = []
        for doc in sample[:3]:
            doc.pop("_id", None)
            examples.append(doc)
        chosen_fields = TEXT_FIELDS or [f for f in field_names if f != "_id"]
        return json.dumps({
            "database": DB_NAME,
            "collection": COLLECTION_NAME,
            "document_count": total,
            "sample_field_names": field_names,
            "configured_text_fields": TEXT_FIELDS,
            "effective_text_fields": chosen_fields,
            "sample_documents": examples,
            "pinecone_index": INDEX_NAME,
            "namespace": NAMESPACE,
            "embedding_model": EMBED_MODEL,
            "embedding_dimension": EMBED_DIMENSION,
        }, default=str, ensure_ascii=False)
    finally:
        client.close()


@tool
def migrate_collection_to_pinecone() -> str:
    """Chunk, embed, and upsert all MongoDB documents to Pinecone; stable IDs make reruns idempotent."""
    client, collection = mongo_collection()
    source_count = collection.count_documents({})
    if source_count == 0:
        client.close()
        return json.dumps({
            "error": f"MongoDB collection {DB_NAME}.{COLLECTION_NAME} is empty. Verify MONGODB_DATABASE and MONGODB_COLLECTION.",
            "source_documents_seen": 0,
            "vectors_upserted": 0,
        })
    index = pinecone_index()
    # Atlas free/shared tiers reject no_cursor_timeout=True. A normal cursor is
    # sufficient here; consume it continuously and let PyMongo manage its batches.
    cursor = collection.find({}).batch_size(BATCH_SIZE)
    total_docs = 0
    uploaded_vectors = 0
    skipped_docs = 0
    pending = []
    try:
        for doc in cursor:
            total_docs += 1
            source_id = str(doc.get("_id"))
            text = select_document_text(doc)
            doc_chunks = chunks(text)
            if not doc_chunks:
                skipped_docs += 1
                continue
            for chunk_no, chunk_text in enumerate(doc_chunks):
                metadata = {
                    "source_database": DB_NAME,
                    "source_collection": COLLECTION_NAME,
                    "mongo_id": source_id,
                    "chunk_index": chunk_no,
                    "text": chunk_text,
                }
                for field in METADATA_FIELDS:
                    value = clean_metadata_value(get_path(doc, field))
                    if value is not None:
                        metadata[field.replace(".", "_")] = value
                pending.append((vector_id(source_id, chunk_no), chunk_text, metadata))
                if len(pending) >= BATCH_SIZE:
                    _upsert_batch(index, pending)
                    uploaded_vectors += len(pending)
                    pending = []
        if pending:
            _upsert_batch(index, pending)
            uploaded_vectors += len(pending)
        return json.dumps({"source_documents_seen": total_docs,
                           "vectors_upserted": uploaded_vectors,
                           "documents_skipped_empty_text": skipped_docs,
                           "namespace": NAMESPACE})
    except Exception as exc:
        return json.dumps({"error": str(exc), "source_documents_seen": total_docs,
                           "vectors_upserted_before_error": uploaded_vectors,
                           "note": "Rerun safely; deterministic vector IDs overwrite prior chunks."})
    finally:
        cursor.close()
        client.close()


def _upsert_batch(index, records):
    texts = [record[1] for record in records]
    vectors = embeddings.embed_documents(texts)
    if vectors and len(vectors[0]) != EMBED_DIMENSION:
        raise ValueError(f"Embedding model returned dimension {len(vectors[0])}; expected {EMBED_DIMENSION}")
    payload = [{"id": record[0], "values": vector, "metadata": record[2]}
               for record, vector in zip(records, vectors)]
    index.upsert(vectors=payload, namespace=NAMESPACE)


@tool
def validate_pinecone_migration() -> str:
    """Compare MongoDB document count to the source metadata IDs represented in Pinecone namespace."""
    client, collection = mongo_collection()
    index = pinecone_index()
    try:
        source_count = collection.count_documents({})
        stats = index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", None)
        if namespaces is None and isinstance(stats, dict):
            namespaces = stats.get("namespaces", {})
        namespace_stats = (namespaces or {}).get(NAMESPACE, {})
        vector_count = getattr(namespace_stats, "vector_count", None)
        if vector_count is None and isinstance(namespace_stats, dict):
            vector_count = namespace_stats.get("vector_count", 0)
        return json.dumps({
            "source_documents": source_count,
            "target_vectors": int(vector_count or 0),
            "namespace": NAMESPACE,
            "note": "Chunking can produce more vectors than source documents; compare the migration report's uploaded_vectors too."
        })
    finally:
        client.close()


TOOLS = [inspect_mongodb_collection, migrate_collection_to_pinecone,
         validate_pinecone_migration]
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an agentic MongoDB Atlas to Pinecone migration orchestrator.
First call inspect_mongodb_collection and review the collection shape. Then call
migrate_collection_to_pinecone. Finally call validate_pinecone_migration. Use tool
results as evidence; never claim success if a tool returned an error. The configured
TEXT_FIELDS control what is embedded; if none are configured, the migration tool uses
all non-_id document fields. Explain chunk counts, skipped empty documents, and any
validation caveat. Do not delete or overwrite unrelated Pinecone namespaces."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent = create_tool_calling_agent(chat, TOOLS, PROMPT)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True,
                         max_iterations=8, handle_parsing_errors=True)

if __name__ == "__main__":
    result = executor.invoke({"input": "Migrate the configured MongoDB Atlas collection to the configured Pinecone index and namespace. Inspect first, migrate, validate, and report counts or errors."})
    print("\n=== MIGRATION REPORT ===\n")
    print(result["output"])
