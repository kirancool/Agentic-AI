import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv
from google.cloud import storage
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
STAGES = ["preflight", "export", "transfer", "dataflow", "validation"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_state(path, state):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(path)


def run_python(script_name):
    command = [sys.executable, "-u", str(ROOT / script_name)]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=os.environ.copy())
    output = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output.append(line)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command, output="".join(output))
    return "".join(output)


def inventory_table():
    table_name = os.environ["DDB_TABLE"]
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError("Set AWS_REGION")
    ddb = boto3.client("dynamodb", region_name=region)
    table = ddb.describe_table(TableName=table_name)["Table"]
    backups = ddb.describe_continuous_backups(TableName=table_name)["ContinuousBackupsDescription"]
    pitr = backups.get("PointInTimeRecoveryDescription", {}).get("PointInTimeRecoveryStatus")
    if table.get("TableStatus") != "ACTIVE":
        raise RuntimeError(f"DynamoDB table must be ACTIVE; got {table.get('TableStatus')}")
    if pitr != "ENABLED":
        raise RuntimeError(f"DynamoDB PITR must be ENABLED before export; got {pitr}")
    required = ["OPENAI_API_KEY", "DDB_EXPORT_BUCKET", "GOOGLE_CLOUD_PROJECT", "GCS_TRANSFER_BUCKET",
                "DATAFLOW_REGION", "DATAFLOW_TEMP_GCS", "DATAFLOW_STAGING_GCS",
                "DATAFLOW_WORKER_SERVICE_ACCOUNT", "DDB_DEAD_LETTER_GCS"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError("Missing required .env settings: " + ", ".join(missing))
    credential_file = Path(os.getenv("AWS_TRANSFER_CREDENTIALS_FILE", "aws-transfer-creds.json"))
    if not credential_file.is_absolute():
        credential_file = ROOT / credential_file
    if not credential_file.exists():
        raise FileNotFoundError(f"Storage Transfer AWS credentials not found: {credential_file}")
    return {
        "table_name": table_name,
        "table_arn": table["TableArn"],
        "table_status": table.get("TableStatus"),
        "pitr_status": pitr,
        "key_schema": table.get("KeySchema", []),
        "attribute_definitions": table.get("AttributeDefinitions", []),
        "global_secondary_indexes": [
            {"name": x["IndexName"], "key_schema": x["KeySchema"], "projection": x.get("Projection")}
            for x in table.get("GlobalSecondaryIndexes", [])
        ],
        "local_secondary_indexes": [
            {"name": x["IndexName"], "key_schema": x["KeySchema"], "projection": x.get("Projection")}
            for x in table.get("LocalSecondaryIndexes", [])
        ],
        "item_count_estimate": table.get("ItemCount"),
        "size_bytes_estimate": table.get("TableSizeBytes"),
    }


def parse_gs_uri(uri):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.rstrip("/")


def count_gcs_objects(uri):
    bucket_name, prefix = parse_gs_uri(uri)
    client = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
    return sum(1 for blob in client.list_blobs(bucket_name, prefix=prefix) if (blob.size or 0) > 0)


def make_stage_action(stage, action, state, state_path):
    def run_stage(input_text: str) -> str:
        current = state.setdefault("stages", {}).get(stage, {})
        if current.get("status") == "complete":
            return current.get("result", "Stage already completed.")
        state["stages"][stage] = {"status": "running", "started_at": utc_now()}
        save_state(state_path, state)
        try:
            result = action(input_text)
            state["stages"][stage] = {"status": "complete", "completed_at": utc_now(), "result": result}
            save_state(state_path, state)
            return result
        except Exception as exc:
            state["stages"][stage] = {"status": "failed", "failed_at": utc_now(), "error": str(exc)}
            save_state(state_path, state)
            raise
    return run_stage


def invoke_stage_agent(client, stage, stage_action, user_input):
    """Run a single GPT-6 Astra specialist with one forced stage-specific tool."""
    tool_name = "run_" + stage + "_stage"
    tool = {
        "type": "function", "name": tool_name,
        "description": f"Execute the deterministic Python {stage} stage and return its actual result.",
        "parameters": {"type": "object", "properties": {"request": {"type": "string"}},
                       "required": ["request"], "additionalProperties": False},
        "strict": True,
    }
    instructions = (f"You are the {stage} specialist agent in a sequential DynamoDB-to-Firestore migration. "
                    f"Call only {tool_name}, exactly once. Do not claim success before receiving the Python result. "
                    "Do not attempt any later stage. The Python controller independently enforces order and success.")
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-astra"), reasoning={"effort": "high"},
        instructions=instructions, input=user_input, tools=[tool],
        tool_choice={"type": "function", "name": tool_name}, parallel_tool_calls=False,
    )
    calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
    if len(calls) != 1 or calls[0].name != tool_name:
        raise RuntimeError(f"{stage} agent did not return its required tool call")
    call = calls[0]
    arguments = json.loads(call.arguments)
    if set(arguments) != {"request"} or not isinstance(arguments["request"], str):
        raise RuntimeError(f"{stage} agent returned invalid arguments")
    print(f"\n=== {stage.upper()} AGENT ===", flush=True)
    result = stage_action(arguments["request"])
    final = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-astra"), reasoning={"effort": "low"},
        instructions=instructions + " Summarize the tool result briefly, without adding claims.",
        previous_response_id=response.id,
        input=[{"type": "function_call_output", "call_id": call.call_id, "output": result}],
        tools=[tool], tool_choice="none", parallel_tool_calls=False,
    )
    print(final.output_text or result, flush=True)


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Sequential AI-agent DynamoDB to Firestore migration")
    parser.add_argument("--new-run", action="store_true", help="start a new export/migration instead of resuming the saved run")
    args = parser.parse_args()
    state_path = Path(os.getenv("MIGRATION_STATE_FILE", str(ROOT / "migration_state.json")))
    if not state_path.is_absolute():
        state_path = ROOT / state_path

    if state_path.exists() and not args.new_run:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            print(f"Run {state['run_id']} is already complete. Use --new-run to start another snapshot.")
            return
        print(f"Resuming migration run {state['run_id']}")
    else:
        run_id = uuid.uuid4().hex[:12]
        state = {"run_id": run_id, "table": os.environ["DDB_TABLE"], "status": "running",
                 "started_at": utc_now(), "stages": {}}
        save_state(state_path, state)
    os.environ["EXPORT_METADATA_FILE"] = str(ROOT / f"migration_{state['run_id']}_export.json")

    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def preflight(_):
        inventory = inventory_table()
        assessment = llm.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-6-astra"), reasoning={"effort": "high"},
            instructions=("You are the DynamoDB-to-Firestore schema review agent. Review only table metadata. "
                          "Assess key-to-document-ID mapping, index/query redesign, type conversion, document-size and "
                          "field-name risks, and snapshot cutover limitations. Do not claim data was migrated."),
            input="Review this DynamoDB metadata inventory:\n" + json.dumps(inventory, default=str),
        )
        state["inventory"] = inventory
        state["risk_assessment"] = assessment.output_text
        save_state(state_path, state)
        print("=== GPT-6 Astra preflight assessment ===\n" + assessment.output_text)
        return json.dumps({"preflight": "passed", "inventory": inventory, "assessment": assessment.output_text})

    def export(_):
        run_python("export_dynamodb_to_s3.py")
        metadata = json.loads(Path(os.environ["EXPORT_METADATA_FILE"]).read_text(encoding="utf-8"))
        return json.dumps({"export_status": metadata["export_status"], "export_arn": metadata["export_arn"],
                           "item_count": metadata.get("item_count"), "key_schema": metadata["key_schema"]})

    def transfer(_):
        run_python("transfer_s3_to_gcs.py")
        metadata = json.loads(Path(os.environ["EXPORT_METADATA_FILE"]).read_text(encoding="utf-8"))
        return json.dumps({"transfer_status": "SUCCESS", "operation": metadata.get("transfer_operation_name"),
                           "input_pattern": metadata["gcs_input_pattern"],
                           "counters": metadata.get("transfer_counters", {})})

    def dataflow(_):
        run_python("run_dataflow.py")
        metadata = json.loads(Path(os.environ["EXPORT_METADATA_FILE"]).read_text(encoding="utf-8"))
        return json.dumps({"job_id": metadata["dataflow_job_id"],
                           "state": metadata["dataflow_final_state"],
                           "metrics": metadata.get("dataflow_metrics", {})})

    def validate(_):
        metadata_path = Path(os.environ["EXPORT_METADATA_FILE"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("item_count")
        metrics = metadata.get("dataflow_metrics", {})
        succeeded = metrics.get("items_succeeded")
        if expected is None or succeeded is None:
            raise RuntimeError(f"Cannot verify export item count against Dataflow successes: expected={expected}, succeeded={succeeded}")
        base = os.environ["DDB_DEAD_LETTER_GCS"].rstrip("/") + "/" + metadata["export_arn"].rsplit("/", 1)[-1]
        parse_errors = count_gcs_objects(base + "/parse-errors")
        write_errors = count_gcs_objects(base + "/firestore-write-errors")
        report = {"expected_export_items": expected, "firestore_items_succeeded": succeeded,
                  "parse_dead_letter_files": parse_errors, "write_dead_letter_files": write_errors,
                  "counts_match": int(expected) == int(succeeded), "job_id": metadata.get("dataflow_job_id")}
        if not report["counts_match"] or parse_errors or write_errors:
            raise RuntimeError("Validation failed: " + json.dumps(report))
        metadata["validation"] = report
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return json.dumps(report)

    actions = {"preflight": preflight, "export": export, "transfer": transfer,
               "dataflow": dataflow, "validation": validate}
    descriptions = {
        "preflight": "Inspect source readiness and assess migration risks.",
        "export": "Create and wait for the DynamoDB point-in-time S3 export.",
        "transfer": "Transfer the S3 export to GCS and wait until the managed transfer finishes.",
        "dataflow": "Run the Python Dataflow loader and wait for its final job state.",
        "validation": "Compare exported item count with successful Firestore writes and check dead letters.",
    }

    for stage in STAGES:
        if state.get("stages", {}).get(stage, {}).get("status") == "complete":
            print(f"Skipping completed stage: {stage}")
            continue
        stage_action = make_stage_action(stage, actions[stage], state, state_path)
        invoke_stage_agent(llm, stage, stage_action,
                           f"Run the {stage} stage for migration {state['run_id']} of DynamoDB table {state['table']}. "
                           f"Stage purpose: {descriptions[stage]} Do not skip the required tool call.")
        if state.get("stages", {}).get(stage, {}).get("status") != "complete":
            raise RuntimeError(f"{stage} agent did not complete its stage tool")

    state["status"] = "complete"
    state["completed_at"] = utc_now()
    save_state(state_path, state)
    print("\nAll sequential AI-agent stages completed successfully.")
    print(f"Final migration report: {Path(os.environ['EXPORT_METADATA_FILE']).resolve()}")


if __name__ == "__main__":
    main()
