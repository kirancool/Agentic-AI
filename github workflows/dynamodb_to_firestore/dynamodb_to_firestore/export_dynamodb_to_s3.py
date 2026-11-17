import json
import os
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv


def main():
    load_dotenv()
    table_name = os.environ["DDB_TABLE"]
    bucket = os.environ["DDB_EXPORT_BUCKET"]
    prefix = os.getenv("DDB_EXPORT_PREFIX", "dynamodb-exports").strip("/")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION")

    client = boto3.client("dynamodb", region_name=region)
    table = client.describe_table(TableName=table_name)["Table"]
    backups = client.describe_continuous_backups(TableName=table_name)["ContinuousBackupsDescription"]
    pitr_status = backups.get("PointInTimeRecoveryDescription", {}).get("PointInTimeRecoveryStatus")
    if pitr_status != "ENABLED":
        raise SystemExit(f"Point-in-time recovery must be ENABLED before export; current status: {pitr_status}")

    output_file = Path(os.getenv("EXPORT_METADATA_FILE", "dynamodb_export_metadata.json"))
    existing = json.loads(output_file.read_text(encoding="utf-8")) if output_file.exists() else {}
    export_arn = existing.get("export_arn") if existing.get("table_name") == table_name else None
    desc = None
    if export_arn:
        desc = client.describe_export(ExportArn=export_arn)["ExportDescription"]
        if desc["ExportStatus"] == "COMPLETED":
            print("Reusing completed DynamoDB export.")
        elif desc["ExportStatus"] in ("FAILED", "CANCELLED"):
            export_arn = None
            desc = None
        else:
            print(f"Resuming DynamoDB export: {export_arn}")

    if not export_arn:
        result = client.export_table_to_point_in_time(
            TableArn=table["TableArn"],
            S3Bucket=bucket,
            S3Prefix=prefix,
            ExportFormat="DYNAMODB_JSON",
            ExportType="FULL_EXPORT",
            S3SseAlgorithm="AES256",
        )
        export_arn = result["ExportDescription"]["ExportArn"]
        export_id = export_arn.rsplit("/", 1)[-1]
        key_schema = sorted(table["KeySchema"], key=lambda k: 0 if k["KeyType"] == "HASH" else 1)
        summary = {
            "table_name": table_name,
            "table_arn": table["TableArn"],
            "key_schema": key_schema,
            "item_count": None,
            "export_arn": export_arn,
            "s3_bucket": bucket,
            "s3_data_prefix": f"{prefix}/AWSDynamoDB/{export_id}/data/",
            "export_status": "IN_PROGRESS",
            "region": region,
        }
        output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Started DynamoDB export: {export_arn}")

    while desc is None or desc["ExportStatus"] != "COMPLETED":
        desc = client.describe_export(ExportArn=export_arn)["ExportDescription"]
        status = desc["ExportStatus"]
        print(f"Export status: {status}")
        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"DynamoDB export ended with status {status}: {desc.get('FailureMessage', '')}")
        if status != "COMPLETED":
            time.sleep(30)

    export_id = export_arn.rsplit("/", 1)[-1]
    key_schema = sorted(table["KeySchema"], key=lambda k: 0 if k["KeyType"] == "HASH" else 1)
    summary = {
        "table_name": table_name,
        "table_arn": table["TableArn"],
        "key_schema": key_schema,
        "item_count": desc.get("ItemCount"),
        "export_arn": export_arn,
        "s3_bucket": bucket,
        "s3_data_prefix": f"{prefix}/AWSDynamoDB/{export_id}/data/",
        "export_status": "COMPLETED",
        "region": region,
        "export_bytes": desc.get("BilledSizeBytes"),
    }
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote export metadata to {output_file.resolve()}")


if __name__ == "__main__":
    main()
