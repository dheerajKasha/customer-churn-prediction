"""
ChurnGuard — S3 New-Data Checker
Determines whether new raw data files have arrived since the last pipeline run.
Writes new_data=true|false to $GITHUB_OUTPUT so the calling workflow can decide
whether to trigger retraining.  Always exits 0 — a check failure is non-fatal.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


EPOCH_ZERO = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check S3 for new raw data files since the last processed marker."
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET_NAME"),
        required=not os.environ.get("S3_BUCKET_NAME"),
        help="S3 bucket to inspect (or set S3_BUCKET_NAME env var)",
    )
    parser.add_argument(
        "--prefix",
        default="data/raw/",
        help="S3 key prefix to scan for new objects (default: data/raw/)",
    )
    parser.add_argument(
        "--marker-key",
        default="data/.last_processed",
        help="S3 key of the last-processed timestamp marker object "
             "(default: data/.last_processed)",
    )
    return parser.parse_args()


def read_marker_timestamp(s3_client, bucket: str, marker_key: str) -> datetime:
    """
    Read the ISO-8601 timestamp stored in the S3 marker object.
    Returns epoch zero if the marker does not exist yet.
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=marker_key)
        raw = response["Body"].read().decode("utf-8").strip()
        ts = datetime.fromisoformat(raw)
        # Ensure timezone-aware
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        print(f"Last-processed marker found: {ts.isoformat()}")
        return ts
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in ("NoSuchKey", "404"):
            print("No last-processed marker found; treating as first run (epoch 0).")
            return EPOCH_ZERO
        # Unexpected error — re-raise so caller can catch it
        raise


def list_new_objects(
    s3_client,
    bucket: str,
    prefix: str,
    since: datetime,
) -> list[dict]:
    """
    Return all S3 objects under *prefix* with LastModified > *since*.
    Handles pagination automatically.
    """
    new_objects: list[dict] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Skip the prefix "folder" placeholder itself
            if obj["Key"].rstrip("/") == prefix.rstrip("/"):
                continue
            last_modified: datetime = obj["LastModified"]
            # Ensure timezone-aware for comparison
            if last_modified.tzinfo is None:
                last_modified = last_modified.replace(tzinfo=timezone.utc)
            if last_modified > since:
                new_objects.append(obj)
    return new_objects


def update_marker(s3_client, bucket: str, marker_key: str, ts: datetime) -> None:
    """Write the current timestamp to the S3 marker object."""
    s3_client.put_object(
        Bucket=bucket,
        Key=marker_key,
        Body=ts.isoformat().encode("utf-8"),
        ContentType="text/plain",
    )
    print(f"Marker updated to {ts.isoformat()}")


def set_github_output(key: str, value: str) -> None:
    """Append key=value to $GITHUB_OUTPUT (no-op when not running in Actions)."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"{key}={value}\n")
    else:
        # For local debugging
        print(f"[github-output] {key}={value}")


def main() -> None:
    args = parse_args()

    try:
        s3_client = boto3.client("s3")

        # 1. Read the last-processed timestamp from the marker object
        last_processed = read_marker_timestamp(s3_client, args.bucket, args.marker_key)

        # 2. List objects modified after that timestamp
        new_objects = list_new_objects(
            s3_client, args.bucket, args.prefix, since=last_processed
        )

        if new_objects:
            print(
                f"Found {len(new_objects)} new file(s) under s3://{args.bucket}/{args.prefix} "
                f"since {last_processed.isoformat()}:"
            )
            for obj in new_objects:
                print(f"  {obj['Key']}  ({obj['LastModified'].isoformat()})")

            # 3a. Update marker to now so next run won't pick up the same files
            now = datetime.now(tz=timezone.utc)
            update_marker(s3_client, args.bucket, args.marker_key, now)

            # 4a. Signal to GitHub Actions that retraining should proceed
            set_github_output("new_data", "true")
        else:
            print(
                f"No new files found under s3://{args.bucket}/{args.prefix} "
                f"since {last_processed.isoformat()}. Skipping retraining."
            )
            # 4b. Signal that retraining should be skipped
            set_github_output("new_data", "false")

    except Exception as exc:  # pylint: disable=broad-except
        # Non-fatal: log the error but exit 0 so the workflow can decide what to do
        print(
            f"WARNING: check_new_data encountered an error: {exc}",
            file=sys.stderr,
        )
        # Default to false so we don't trigger an unnecessary pipeline run
        set_github_output("new_data", "false")

    sys.exit(0)


if __name__ == "__main__":
    main()
