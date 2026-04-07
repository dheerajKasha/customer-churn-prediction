"""
approve_model.py

Purpose: Approve the latest pending model in SageMaker Model Registry.
Used by the deploy.yml GitHub Actions workflow after the manual approval gate.
"""

import argparse
import os
import sys

import boto3
from botocore.exceptions import ClientError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approve the latest PendingManualApproval model in SageMaker Model Registry."
    )
    parser.add_argument(
        "--model-package-group",
        default="ChurnGuardModels",
        help="SageMaker Model Package Group name (default: ChurnGuardModels)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    return parser.parse_args()


def publish_sns(sns_client, topic_arn, message, subject):
    try:
        sns_client.publish(
            TopicArn=topic_arn,
            Message=message,
            Subject=subject,
        )
        print(f"SNS notification sent: {subject}")
    except ClientError as e:
        print(f"Warning: failed to publish SNS notification — {e.response['Error']['Message']}")


def main():
    args = parse_args()

    try:
        sm_client = boto3.client("sagemaker", region_name=args.region)
    except Exception as e:
        print(f"Error: could not create SageMaker boto3 client — {e}")
        sys.exit(1)

    # List models pending manual approval, most recently created first
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=args.model_package_group,
            ModelApprovalStatus="PendingManualApproval",
            SortBy="CreationTime",
            SortOrder="Descending",
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code == "ValidationException":
            print(
                f"Error: model package group '{args.model_package_group}' not found or invalid. "
                f"Check that the group exists in region {args.region}.\nDetails: {msg}"
            )
        elif code == "AccessDeniedException":
            print(
                f"Error: insufficient permissions to list model packages. "
                f"Ensure the IAM role has 'sagemaker:ListModelPackages'.\nDetails: {msg}"
            )
        else:
            print(f"Error calling list_model_packages [{code}]: {msg}")
        sys.exit(1)

    packages = response.get("ModelPackageSummaryList", [])

    if not packages:
        print(
            f"Warning: no models with status 'PendingManualApproval' found in group "
            f"'{args.model_package_group}'. Nothing to approve."
        )
        sys.exit(1)

    # Take the latest (first after descending sort)
    latest = packages[0]
    arn = latest["ModelPackageArn"]

    print(f"Found pending model: {arn}")
    print(f"Created: {latest.get('CreationTime', 'unknown')}")

    # Approve the model
    try:
        sm_client.update_model_package(
            ModelPackageArn=arn,
            ModelApprovalStatus="Approved",
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code == "AccessDeniedException":
            print(
                f"Error: insufficient permissions to update model package. "
                f"Ensure the IAM role has 'sagemaker:UpdateModelPackage'.\nDetails: {msg}"
            )
        else:
            print(f"Error calling update_model_package [{code}]: {msg}")
        sys.exit(1)

    print(f"Approved: {arn}")

    # Optional SNS notification
    topic_arn = os.environ.get("SNS_TOPIC_ARN")
    if topic_arn:
        sns_client = boto3.client("sns", region_name=args.region)
        publish_sns(
            sns_client,
            topic_arn,
            message=f"ChurnGuard: Model approved {arn}",
            subject="Model Approved",
        )


if __name__ == "__main__":
    main()
