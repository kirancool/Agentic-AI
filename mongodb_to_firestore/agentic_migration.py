"""Sequential, tool-calling GPT-6 Astra agents for the migration stages."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

import migration


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
        "Ask GPT-6 Astra to review only DocumentDB collection/index metadata, and save the report to GCS. Never read or send document values.",
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
    """Let the stage's Astra agent call only its one approved Python action."""
    tool = {
        "type": "function",
        "name": stage.function_name,
        "description": stage.instruction,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    }
    instructions = (
        f"You are the {stage.name} in a sequential AWS DocumentDB to Google Firestore migration. "
        "You may call only the single Python function provided for your stage. Call it exactly once with no arguments. "
        "Do not invent results, change the migration configuration, or claim success before receiving the function result. "
        "The Python controller independently verifies completion and decides whether the next stage may start."
    )
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-astra"),
        reasoning={"effort": "high"},
        instructions=instructions,
        input=f"Execute your assigned stage. Previous stage result: {previous_summary or 'This is the first stage.'}",
        tools=[tool],
        tool_choice={"type": "function", "name": stage.function_name},
        parallel_tool_calls=False,
    )
    calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
    if len(calls) != 1 or calls[0].name != stage.function_name:
        raise RuntimeError(f"{stage.name} did not return its required function call")
    call = calls[0]
    if json.loads(call.arguments or "{}") != {}:
        raise RuntimeError(f"{stage.name} returned unexpected arguments")

    print(f"\n=== {stage.name}: running Python stage ===", flush=True)
    stage.action()  # exceptions stop the workflow before the next agent starts
    tool_result = json.dumps({"stage": stage.name, "status": "completed"})

    # Complete the Responses function-calling cycle and capture the agent's handoff summary.
    summary = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-astra"),
        reasoning={"effort": "low"},
        instructions=instructions + " Summarize the returned stage result in one short sentence. Do not make claims beyond that result.",
        previous_response_id=response.id,
        input=[{"type": "function_call_output", "call_id": call.call_id, "output": tool_result}],
        tools=[tool],
        tool_choice="none",
        parallel_tool_calls=False,
    )
    output = summary.output_text.strip() if summary.output_text else tool_result
    print(f"{stage.name} handoff: {output}", flush=True)
    return output


def main() -> int:
    if len(sys.argv) != 1:
        print("Usage: python agentic_migration.py", file=sys.stderr)
        return 2
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    previous_summary = ""
    for stage in STAGES:
        previous_summary = invoke_stage_agent(client, stage, previous_summary)
    print("\nMigration workflow completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
