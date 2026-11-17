import argparse
import base64
import hashlib
import json
import math
from decimal import Decimal
from queue import Queue, Empty

import apache_beam as beam
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions, SetupOptions
from apache_beam.runners.runner import PipelineState
from apache_beam.transforms.window import GlobalWindow
from apache_beam.utils.timestamp import MAX_TIMESTAMP
from apache_beam.utils.windowed_value import WindowedValue
from google.cloud import firestore
from google.cloud.firestore_v1.bulk_writer import BulkWriterOptions
from apache_beam.metrics.metric import Metrics
from apache_beam import pvalue


def number_value(raw):
    number = Decimal(raw)
    try:
        return int(number.to_integral_exact()) if -(2**63) <= int(number) < 2**63 and number == number.to_integral_value() else _decimal_float_or_string(number, raw)
    except (OverflowError, ValueError):
        return _decimal_float_or_string(number, raw)


def _decimal_float_or_string(number, raw):
    value = float(number)
    if math.isfinite(value) and Decimal(str(value)) == number:
        return value
    # Firestore supports int64/double, not arbitrary precision DynamoDB numbers.
    return raw


def decode_attribute(attribute):
    if not isinstance(attribute, dict) or len(attribute) != 1:
        raise ValueError("Invalid DynamoDB AttributeValue")
    kind, value = next(iter(attribute.items()))
    if kind == "S":
        return value
    if kind == "N":
        return number_value(value)
    if kind == "B":
        return base64.b64decode(value)
    if kind == "BOOL":
        return bool(value)
    if kind == "NULL":
        return None
    if kind == "M":
        return decode_map(value)
    if kind == "L":
        values = [decode_attribute(v) for v in value]
        return [{"ddb_type": "L", "items": v} if isinstance(v, list) else v for v in values]
    if kind == "SS":
        return list(value)
    if kind == "NS":
        return [number_value(v) for v in value]
    if kind == "BS":
        return [base64.b64decode(v) for v in value]
    raise ValueError(f"Unsupported DynamoDB type: {kind}")


def decode_map(attributes):
    output = {}
    for name, value in attributes.items():
        if name.startswith("__") and name.endswith("__"):
            raise ValueError(f"Firestore-reserved field name: {name}")
        output[name] = decode_attribute(value)
    return output


def document_id(item, partition_key, sort_key):
    names = [partition_key] + ([sort_key] if sort_key else [])
    keys = {name: item[name] for name in names}
    canonical = json.dumps(keys, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = "k_" + base64.urlsafe_b64encode(canonical).decode("ascii").rstrip("=")
    if len(encoded.encode("utf-8")) <= 1500:
        return encoded
    return "h_" + base64.urlsafe_b64encode(hashlib.sha256(canonical).digest()).decode("ascii").rstrip("=")


class ImportItems(beam.DoFn):
    def __init__(self, project, database, collection, partition_key, sort_key, retry_limit,
                 initial_write_rate, max_write_rate):
        self.project = project
        self.database_id = database
        self.collection = collection
        self.partition_key = partition_key
        self.sort_key = sort_key or ""
        self.retry_limit = retry_limit
        self.initial_write_rate = initial_write_rate
        self.max_write_rate = max_write_rate
        self.items_written = Metrics.counter(self.__class__, "items_queued")
        self.items_succeeded = Metrics.counter(self.__class__, "items_succeeded")
        self.parse_failures = Metrics.counter(self.__class__, "parse_failures")
        self.write_failures = Metrics.counter(self.__class__, "write_failures")

    def setup(self):
        self.client = firestore.Client(project=self.project, database=self.database_id)

    def start_bundle(self):
        self.errors = Queue()
        self.writer = self.client.bulk_writer(options=BulkWriterOptions(
            initial_ops_per_second=self.initial_write_rate,
            max_ops_per_second=self.max_write_rate,
        ))
        self.writer.on_write_error(self._on_write_error)
        self.writer.on_write_result(self._on_write_success)

    def _on_write_success(self, reference, result, writer):
        self.items_succeeded.inc()

    def _on_write_error(self, failure, writer):
        attempts = failure.attempts
        if attempts < self.retry_limit:
            return True
        operation = failure.operation
        reference = getattr(operation, "reference", None)
        self.errors.put(json.dumps({
            "document": getattr(reference, "path", "unknown"),
            "error_code": failure.code,
            "error": failure.message,
            "attempts": attempts,
        }))
        self.write_failures.inc()
        return False

    def process(self, line):
        if not line.strip():
            return
        try:
            row = json.loads(line)
            item = row["Item"]
            if not isinstance(item, dict):
                raise ValueError("Export row has no Item map")
            if self.partition_key not in item or (self.sort_key and self.sort_key not in item):
                raise ValueError("DynamoDB key field is missing")
            doc_id = document_id(item, self.partition_key, self.sort_key)
            payload = decode_map(item)
            self.writer.set(self.client.collection(self.collection).document(doc_id), payload)
            self.items_written.inc()
        except Exception as exc:
            self.parse_failures.inc()
            yield pvalue.TaggedOutput("parse_errors", json.dumps({"error": str(exc), "line": line}, ensure_ascii=False))

    def finish_bundle(self):
        self.writer.close()
        try:
            while True:
                error = self.errors.get_nowait()
                yield WindowedValue(pvalue.TaggedOutput("write_errors", error), MAX_TIMESTAMP, (GlobalWindow(),))
        except Empty:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pattern", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--firestore-database", default="(default)")
    parser.add_argument("--partition-key", required=True)
    parser.add_argument("--sort-key", default="")
    parser.add_argument("--dead-letter-prefix", required=True)
    parser.add_argument("--retry-limit", type=int, default=5)
    parser.add_argument("--initial-write-rate", type=int, default=50)
    parser.add_argument("--max-write-rate", type=int, default=500)
    parser.add_argument("--requirements-file", default="requirements-dataflow.txt")
    args, beam_args = parser.parse_known_args()

    options = PipelineOptions(beam_args)
    setup_options = options.view_as(SetupOptions)
    setup_options.requirements_file = args.requirements_file
    setup_options.save_main_session = True
    pipeline = beam.Pipeline(options=options)
    outputs = (
        pipeline
        | "ReadDynamoDBExport" >> beam.io.ReadFromText(
            args.input_pattern, compression_type=CompressionTypes.AUTO)
        | "ConvertAndWriteFirestore" >> beam.ParDo(
            ImportItems(options.view_as(GoogleCloudOptions).project,
                        args.firestore_database, args.collection, args.partition_key,
                        args.sort_key, args.retry_limit, args.initial_write_rate,
                        args.max_write_rate)
        ).with_outputs("parse_errors", "write_errors")
    )
    (outputs.parse_errors
     | "WriteParseDeadLetter" >> beam.io.WriteToText(
         args.dead_letter_prefix.rstrip("/") + "/parse-errors", file_name_suffix=".jsonl"))
    (outputs.write_errors
     | "WriteFirestoreDeadLetter" >> beam.io.WriteToText(
         args.dead_letter_prefix.rstrip("/") + "/firestore-write-errors", file_name_suffix=".jsonl"))
    result = pipeline.run()
    final_state = result.wait_until_finish()
    if final_state != PipelineState.DONE:
        raise RuntimeError(f"Dataflow pipeline ended in state {final_state}")
    print(f"DATAFLOW_JOB_ID={result.job_id()}", flush=True)
    print(f"DATAFLOW_FINAL_STATE={final_state}", flush=True)


if __name__ == "__main__":
    main()
