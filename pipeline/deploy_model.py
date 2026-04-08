"""
deploy_model.py

Purpose: Deploy the latest approved model to a SageMaker real-time endpoint.
Handles both initial endpoint creation and rolling updates of an existing endpoint.
Used by the deploy.yml GitHub Actions workflow.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import boto3
from botocore.exceptions import ClientError


# Maximum time (seconds) to wait for the endpoint to reach InService/Failed
ENDPOINT_WAIT_TIMEOUT = 900  # 15 minutes
POLL_INTERVAL = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy the latest Approved model from SageMaker Model Registry to an endpoint."
    )
    parser.add_argument(
        "--model-package-group",
        default="ChurnGuardModels",
        help="SageMaker Model Package Group name (default: ChurnGuardModels)",
    )
    parser.add_argument(
        "--endpoint-name",
        default="churnguard-endpoint",
        help="SageMaker endpoint name (default: churnguard-endpoint)",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.m5.large",
        help="SageMaker instance type (default: ml.m5.large)",
    )
    parser.add_argument(
        "--initial-instance-count",
        type=int,
        default=1,
        help="Number of instances for the endpoint (default: 1)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    return parser.parse_args()


def get_sagemaker_role():
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role_arn:
        raise EnvironmentError(
            "Required environment variable SAGEMAKER_ROLE_ARN is not set. "
            "Set it to the IAM role ARN that SageMaker should use to run the endpoint."
        )
    return role_arn


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


def get_latest_approved_model(sm_client, model_package_group):
    """Return the ARN of the most recently approved model package, or raise."""
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code == "ValidationException":
            raise RuntimeError(
                f"Model package group '{model_package_group}' not found or invalid. "
                f"Check that the group exists.\nDetails: {msg}"
            )
        elif code == "AccessDeniedException":
            raise RuntimeError(
                f"Insufficient permissions to list model packages. "
                f"Ensure the IAM role has 'sagemaker:ListModelPackages'.\nDetails: {msg}"
            )
        else:
            raise RuntimeError(f"Error calling list_model_packages [{code}]: {msg}")

    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(
            f"No Approved models found in group '{model_package_group}'. "
            "Run the pipeline and approve a model first."
        )

    latest = packages[0]
    arn = latest["ModelPackageArn"]
    print(f"Latest approved model package: {arn}")
    print(f"Created: {latest.get('CreationTime', 'unknown')}")
    return arn


def create_sagemaker_model(sm_client, model_name, model_package_arn, role_arn):
    """Create a SageMaker Model resource from a Model Registry package ARN."""
    print(f"Creating SageMaker model: {model_name}")
    try:
        sm_client.create_model(
            ModelName=model_name,
            Containers=[
                {
                    "ModelPackageName": model_package_arn,
                }
            ],
            ExecutionRoleArn=role_arn,
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code == "AccessDeniedException":
            raise RuntimeError(
                f"Insufficient permissions to create model. "
                f"Ensure the IAM role has 'sagemaker:CreateModel'.\nDetails: {msg}"
            )
        else:
            raise RuntimeError(f"Error calling create_model [{code}]: {msg}")
    print(f"Model created: {model_name}")


def create_endpoint_config(sm_client, config_name, model_name, instance_type, instance_count):
    """Create a SageMaker EndpointConfig with a single production variant."""
    print(f"Creating endpoint config: {config_name}")
    try:
        sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": instance_count,
                    "InitialVariantWeight": 1.0,
                }
            ],
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code == "AccessDeniedException":
            raise RuntimeError(
                f"Insufficient permissions to create endpoint config. "
                f"Ensure the IAM role has 'sagemaker:CreateEndpointConfig'.\nDetails: {msg}"
            )
        else:
            raise RuntimeError(f"Error calling create_endpoint_config [{code}]: {msg}")
    print(f"Endpoint config created: {config_name}")


def endpoint_exists(sm_client, endpoint_name):
    """Return True if an endpoint with the given name already exists."""
    try:
        response = sm_client.list_endpoints(
            NameContains=endpoint_name,
            MaxResults=10,
        )
        for ep in response.get("Endpoints", []):
            if ep["EndpointName"] == endpoint_name:
                return True
        return False
    except ClientError as e:
        raise RuntimeError(
            f"Error checking endpoint existence: {e.response['Error']['Message']}"
        )


def create_or_update_endpoint(sm_client, endpoint_name, config_name):
    """Create the endpoint if it doesn't exist, otherwise update it."""
    if endpoint_exists(sm_client, endpoint_name):
        print(f"Endpoint '{endpoint_name}' already exists — updating.")
        try:
            sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = e.response["Error"]["Message"]
            if code == "AccessDeniedException":
                raise RuntimeError(
                    f"Insufficient permissions to update endpoint. "
                    f"Ensure the IAM role has 'sagemaker:UpdateEndpoint'.\nDetails: {msg}"
                )
            else:
                raise RuntimeError(f"Error calling update_endpoint [{code}]: {msg}")
        print(f"Endpoint update initiated: {endpoint_name}")
    else:
        print(f"Endpoint '{endpoint_name}' does not exist — creating.")
        try:
            sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = e.response["Error"]["Message"]
            if code == "AccessDeniedException":
                raise RuntimeError(
                    f"Insufficient permissions to create endpoint. "
                    f"Ensure the IAM role has 'sagemaker:CreateEndpoint'.\nDetails: {msg}"
                )
            else:
                raise RuntimeError(f"Error calling create_endpoint [{code}]: {msg}")
        print(f"Endpoint creation initiated: {endpoint_name}")


def poll_endpoint_status(sm_client, endpoint_name):
    """
    Poll the endpoint until it reaches InService or Failed.
    Returns the final status string.
    """
    print(f"Waiting for endpoint '{endpoint_name}' to become InService", end="", flush=True)
    deadline = time.time() + ENDPOINT_WAIT_TIMEOUT

    while time.time() < deadline:
        try:
            response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            raise RuntimeError(
                f"Error describing endpoint: {e.response['Error']['Message']}"
            )

        status = response["EndpointStatus"]

        if status == "InService":
            print(" InService")
            return status, None
        elif status == "Failed":
            failure_reason = response.get("FailureReason", "No failure reason provided.")
            print(" Failed")
            return status, failure_reason

        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)

    # Timed out
    print(" Timed out")
    return "TimedOut", f"Endpoint did not reach InService within {ENDPOINT_WAIT_TIMEOUT} seconds."


def print_invocation_example(endpoint_name, region):
    """Print a boto3 sagemaker-runtime example for invoking the endpoint."""
    print("\n--- Invocation example (boto3) ---")
    print(f"""
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="{region}")

# Replace with actual feature values in CSV order
payload = "0,12,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,29.85"

response = runtime.invoke_endpoint(
    EndpointName="{endpoint_name}",
    ContentType="text/csv",
    Body=payload,
)

prediction = json.loads(response["Body"].read().decode())
print("Churn probability:", prediction)
""")
    print("----------------------------------")


def main():
    args = parse_args()

    try:
        role_arn = get_sagemaker_role()
    except EnvironmentError as e:
        print(f"Error: {e}")
        sys.exit(1)

    sm_client = boto3.client("sagemaker", region_name=args.region)
    sns_topic_arn = os.environ.get("SNS_TOPIC_ARN")
    sns_client = boto3.client("sns", region_name=args.region) if sns_topic_arn else None

    # --- Step 1: Find latest approved model ---
    try:
        model_package_arn = get_latest_approved_model(sm_client, args.model_package_group)
    except RuntimeError as e:
        print(f"Error: {e}")
        if sns_client:
            publish_sns(
                sns_client,
                sns_topic_arn,
                message=f"ChurnGuard deployment FAILED: {e}",
                subject="ChurnGuard Deployment Failed",
            )
        sys.exit(1)

    # --- Step 2: Generate unique resource names ---
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_name = f"churnguard-{timestamp}"
    config_name = f"churnguard-config-{timestamp}"

    # --- Step 3: Create SageMaker Model ---
    try:
        create_sagemaker_model(sm_client, model_name, model_package_arn, role_arn)
    except RuntimeError as e:
        print(f"Error: {e}")
        if sns_client:
            publish_sns(
                sns_client,
                sns_topic_arn,
                message=f"ChurnGuard deployment FAILED during model creation: {e}",
                subject="ChurnGuard Deployment Failed",
            )
        sys.exit(1)

    # --- Step 4: Create EndpointConfig ---
    try:
        create_endpoint_config(
            sm_client,
            config_name,
            model_name,
            args.instance_type,
            args.initial_instance_count,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        if sns_client:
            publish_sns(
                sns_client,
                sns_topic_arn,
                message=f"ChurnGuard deployment FAILED during endpoint config creation: {e}",
                subject="ChurnGuard Deployment Failed",
            )
        sys.exit(1)

    # --- Step 5: Create or update endpoint ---
    try:
        create_or_update_endpoint(sm_client, args.endpoint_name, config_name)
    except RuntimeError as e:
        print(f"Error: {e}")
        if sns_client:
            publish_sns(
                sns_client,
                sns_topic_arn,
                message=f"ChurnGuard deployment FAILED during endpoint create/update: {e}",
                subject="ChurnGuard Deployment Failed",
            )
        sys.exit(1)

    # --- Step 6: Poll until InService or Failed ---
    try:
        status, failure_reason = poll_endpoint_status(sm_client, args.endpoint_name)
    except RuntimeError as e:
        print(f"Error: {e}")
        if sns_client:
            publish_sns(
                sns_client,
                sns_topic_arn,
                message=f"ChurnGuard deployment FAILED while polling endpoint: {e}",
                subject="ChurnGuard Deployment Failed",
            )
        sys.exit(1)

    # --- Step 7: Report outcome ---
    if status == "InService":
        print(f"\nEndpoint '{args.endpoint_name}' is InService.")
        print_invocation_example(args.endpoint_name, args.region)

        if sns_client:
            publish_sns(
                sns_client,
                sns_topic_arn,
                message=(
                    f"ChurnGuard: Endpoint '{args.endpoint_name}' deployed successfully.\n"
                    f"Model package: {model_package_arn}\n"
                    f"Instance type: {args.instance_type} x{args.initial_instance_count}"
                ),
                subject="ChurnGuard Deployment Succeeded",
            )
    else:
        # Failed or TimedOut
        print(f"\nDeployment failed. Reason: {failure_reason}")
        if sns_client:
            publish_sns(
                sns_client,
                sns_topic_arn,
                message=(
                    f"ChurnGuard deployment FAILED.\n"
                    f"Endpoint: {args.endpoint_name}\n"
                    f"Status: {status}\n"
                    f"Reason: {failure_reason}"
                ),
                subject="ChurnGuard Deployment Failed",
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
