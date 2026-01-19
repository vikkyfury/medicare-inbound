# medicare-inbound/infra/lambda/kb_auto_sync.py
"""
Auto-sync Bedrock Knowledge Base when new files land in S3.

Env vars to set on the Lambda function:
- KB_ID            (e.g., "ZABCDEF123...")
- DATA_SOURCE_ID   (e.g., "DQY2LX0EDP")
- KB_PREFIX        (optional; default "kb/")
- AWS_REGION       (optional; falls back to Lambda's region)

Trigger: S3 event notification (PUT + CompleteMultipartUpload) for prefix "kb/".
"""

import os
import json
import boto3
from urllib.parse import unquote_plus

KB_ID = os.environ.get("BEDROCK_KNOWLEDGE_BASE_ID")                 # REQUIRED
DATA_SOURCE_ID = os.environ.get("BEDROCK_DATA_SOURCE_ID")  # REQUIRED
KB_PREFIX = os.environ.get("KB_PREFIX", "kb/")  # OPTIONAL

# Correct control-plane client for Knowledge Bases
agent = boto3.client("bedrock-agent")  # NOT "bedrock-agent-runtime"

def _is_kb_object(key: str) -> bool:
    """Return True only for relevant keys under KB_PREFIX and supported types."""
    if not key or not key.startswith(KB_PREFIX):
        return False
    k = key.lower()
    return k.endswith(".md") or k.endswith(".pdf") or k.endswith(".metadata.json")

def lambda_handler(event, context):
    print("Received event:", json.dumps(event, ensure_ascii=False))

    if not KB_ID or not DATA_SOURCE_ID:
        raise RuntimeError("Missing env vars: KB_ID and/or DATA_SOURCE_ID")

    # If any record in the batch is relevant, trigger one ingestion job.
    should_sync = False
    for rec in event.get("Records", []):
        try:
            key = unquote_plus(rec["s3"]["object"]["key"])
        except Exception:
            key = ""
        print(f"[AUTO-SYNC] Saw object key: {key}")
        if _is_kb_object(key):
            should_sync = True

    if not should_sync:
        print("[AUTO-SYNC] No relevant keys under KB_PREFIX; skipping sync.")
        return {"status": "skipped"}

    # Start a new ingestion job
    resp = agent.start_ingestion_job(
        knowledgeBaseId=KB_ID,
        dataSourceId=DATA_SOURCE_ID
    )
    print("[AUTO-SYNC] start_ingestion_job response:", json.dumps(resp, default=str))
    return {"status": "ingestion_started", "job": resp.get("ingestionJob", {})}
