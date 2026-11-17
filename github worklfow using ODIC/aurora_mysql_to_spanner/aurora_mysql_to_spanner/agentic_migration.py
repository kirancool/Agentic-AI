"""Run constrained GPT-6 Astra stage agents sequentially for Aurora -> Spanner."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI
from dotenv import load_dotenv
import migration


@dataclass(frozen=True)
class AgentStage:
    label: str
    function_name: str
    action: Callable[[], None]
    instruction: str


STAGES = [
    AgentStage("Aurora preflight agent", "run_preflight", migration.preflight, "Check Aurora, Spanner, GCS, and source schema readiness."),
    AgentStage("Astra schema review agent", "run_schema_review", migration.ai_review, "Review metadata only and save the schema risk report."),
    AgentStage("Spanner schema agent", "run_schema_setup", migration.ensure_schema, "Generate and, if enabled, apply additive Spanner table DDL; verify target columns."),
    AgentStage("Aurora export agent", "run_aurora_export", migration.export_to_gcs, "Stream source rows into bounded gzip JSONL files in GCS."),
    AgentStage("Dataflow load agent", "run_dataflow_load", migration.load_to_spanner, "Run and wait for the Python Dataflow load into Cloud Spanner."),
    AgentStage("Validation agent", "run_validation", migration.validate, "Compare exported row count with Spanner writes and require empty dead letters."),
]


def execute_agent(client: OpenAI, stage: AgentStage, prior: str) -> str:
    tool = {
        "type": "function", "name": stage.function_name,
        "description": stage.instruction,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    }
    instructions = (
        f"You are the {stage.label} in a sequential Amazon Aurora MySQL to Cloud Spanner migration. "
        f"Call only the supplied function {stage.function_name}, exactly once, with no arguments. "
        "Do not claim success before the Python function returns. Do not perform a later stage. "
        "The Python controller enforces ordering and determines whether the next agent may run."
    )
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-astra"), reasoning={"effort": "high"},
        instructions=instructions,
        input=f"Perform this stage. {stage.instruction} Previous agent result: {prior or 'None; this is the first stage.'}",
        tools=[tool], tool_choice={"type": "function", "name": stage.function_name}, parallel_tool_calls=False,
    )
    calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
    if len(calls) != 1 or calls[0].name != stage.function_name or json.loads(calls[0].arguments or "{}") != {}:
        raise RuntimeError(f"{stage.label} failed to request its required stage tool")
    print(f"\n=== {stage.label} ===", flush=True)
    stage.action()  # Any exception stops the migration before the next agent.
    result = json.dumps({"stage": stage.label, "status": "completed"})
    handoff = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-astra"), reasoning={"effort": "low"},
        instructions=instructions + " Summarize the tool result in one sentence without adding facts.",
        previous_response_id=response.id,
        input=[{"type": "function_call_output", "call_id": calls[0].call_id, "output": result}],
        tools=[tool], tool_choice="none", parallel_tool_calls=False,
    )
    summary = handoff.output_text or result
    print(summary, flush=True)
    return summary


def main() -> None:
    from pathlib import Path
    load_dotenv(Path(__file__).with_name(".env"))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prior = ""
    for stage in STAGES:
        prior = execute_agent(client, stage, prior)
    print("All sequential GPT-6 Astra migration agents completed.", flush=True)


if __name__ == "__main__":
    main()
