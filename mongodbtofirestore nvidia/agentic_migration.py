"""Sequential, tool-calling NVIDIA Nemotron agents for migration stages."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

import migration


MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")


@dataclass(frozen=True)
class StageAgent:
    name: str
    function_name: str
    instruction: str
    action: Callable[[], None]


STAGES = [
    StageAgent(
        "Preflight Agent", "run_preflight",
        "Run the deterministic preflight checks for the configured Amazon DocumentDB source and Google Cloud destination.",
        migration.preflight,
    ),
    StageAgent(
        "Schema Review Agent", "run_schema_review",
        "Ask NVIDIA Nemotron to review only DocumentDB collection/index metadata, and save the report to GCS. Never read or send document values.",
        migration.ai_review,
    ),
    StageAgent(
        "Export Agent", "run_export",
        "Export the configured DocumentDB database to compressed JSONL objects in the run's GCS prefix.",
        migration.export_to_gcs,
    ),
    StageAgent(
        "Dataflow Load Agent", "run_dataflow_load",
        "Run the Python Dataflow pipeline that imports the GCS snapshot into Firestore and wait for it to finish.",
        migration.load_to_firestore,
    ),
    StageAgent(
        "Validation Agent", "run_validation",
        "Validate export counts, Firestore successful writes, and dead-letter output. Report failure if any check fails.",
        migration.validate,
    ),
]


def invoke_stage_agent(client: OpenAI, stage: StageAgent, previous_summary: str) -> str:
    """Ask the model to select its one allowed tool, then run the Python stage."""
    tool = {
        "type": "function",
        "function": {
            "name": stage.function_name,
            "description": stage.instruction,
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    }
    system = (
        f"You are the {stage.name} in a sequential AWS DocumentDB to Google Firestore migration. "
        "Call the only provided function exactly once with no arguments. Do not invent results or claim success. "
        "The Python controller verifies completion and decides whether the next stage may start."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Execute your assigned stage. Previous stage result: {previous_summary or 'This is the first stage.'}"},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": stage.function_name}},
        parallel_tool_calls=False,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    message = response.choices[0].message
    calls = message.tool_calls or []
    if len(calls) != 1 or calls[0].function.name != stage.function_name:
        raise RuntimeError(f"{stage.name} did not return its required function call")
    if json.loads(calls[0].function.arguments or "{}") != {}:
        raise RuntimeError(f"{stage.name} returned unexpected arguments")

    print(f"\n=== {stage.name}: running Python stage ===", flush=True)
    stage.action()  # Exceptions stop the workflow before the next agent starts.
    tool_result = json.dumps({"stage": stage.name, "status": "completed"})
    summary_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system + " Summarize the returned stage result in one short sentence. Do not claim more than the result supports."},
            {"role": "user", "content": "Execute your assigned stage."},
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{"id": calls[0].id, "type": "function", "function": {"name": stage.function_name, "arguments": calls[0].function.arguments or "{}"}}],
            },
            {"role": "tool", "tool_call_id": calls[0].id, "content": tool_result},
        ],
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    output = summary_response.choices[0].message.content or tool_result
    print(f"{stage.name} handoff: {output.strip()}", flush=True)
    return output.strip()


def main() -> int:
    if len(sys.argv) != 1:
        print("Usage: python agentic_migration.py", file=sys.stderr)
        return 2
    client = OpenAI(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    )
    previous_summary = ""
    for stage in STAGES:
        previous_summary = invoke_stage_agent(client, stage, previous_summary)
    print("\nMigration workflow completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
