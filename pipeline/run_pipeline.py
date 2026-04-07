"""
ChurnGuard — Pipeline Execution Runner
Triggers a SageMaker Pipeline execution and optionally polls until completion.
Exits 0 on success, 1 on failure — designed for use inside GitHub Actions.
"""

import argparse
import os
import sys
import time

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline


POLL_INTERVAL_SECONDS = 30

# Terminal execution statuses
TERMINAL_STATUSES = {"Succeeded", "Failed", "Stopped"}
SUCCESS_STATUSES = {"Succeeded"}
FAILURE_STATUSES = {"Failed", "Stopped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trigger a ChurnGuard SageMaker Pipeline execution."
    )
    parser.add_argument(
        "--pipeline-name",
        default=os.environ.get("PIPELINE_NAME", "ChurnGuardPipeline"),
        help="Name of the SageMaker Pipeline (default: ChurnGuardPipeline)",
    )
    parser.add_argument(
        "--input-data-url",
        default=os.environ.get("S3_INPUT_URL"),
        help="S3 URL for raw input data (or set S3_INPUT_URL env var)",
    )
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("SAGEMAKER_ROLE_ARN"),
        help="SageMaker execution role ARN (or set SAGEMAKER_ROLE_ARN env var)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block and poll until the execution reaches a terminal state",
    )
    return parser.parse_args()


def get_step_statuses(sm_client, execution_arn: str) -> list[dict]:
    """Return a list of {StepName, StepStatus} dicts for the given execution."""
    response = sm_client.list_pipeline_execution_steps(
        PipelineExecutionArn=execution_arn
    )
    return [
        {
            "StepName": step["StepName"],
            "StepStatus": step["StepStatus"],
        }
        for step in response.get("PipelineExecutionSteps", [])
    ]


def poll_until_complete(sm_client, execution_arn: str) -> str:
    """
    Poll the execution every POLL_INTERVAL_SECONDS seconds.
    Prints step statuses on each poll.
    Returns the final PipelineExecutionStatus string.
    """
    print(f"\nPolling execution: {execution_arn}")
    print(f"Polling every {POLL_INTERVAL_SECONDS}s …\n")

    while True:
        response = sm_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
        status = response["PipelineExecutionStatus"]

        step_statuses = get_step_statuses(sm_client, execution_arn)
        print(f"[{time.strftime('%H:%M:%S')}] Execution status: {status}")
        for step in step_statuses:
            print(f"  {step['StepName']:40s}  {step['StepStatus']}")
        print()

        if status in TERMINAL_STATUSES:
            return status

        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> None:
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_client = boto_session.client("sagemaker")
    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
    )

    # Build execution parameters (only pass overrides for non-default values)
    execution_params: list[dict] = []
    if args.input_data_url:
        execution_params.append(
            {"Name": "InputDataUrl", "Value": args.input_data_url}
        )

    # Start execution
    start_kwargs: dict = {
        "PipelineName": args.pipeline_name,
        "PipelineExecutionDisplayName": f"github-actions-{int(time.time())}",
    }
    if execution_params:
        start_kwargs["PipelineParameters"] = execution_params

    response = sm_client.start_pipeline_execution(**start_kwargs)
    execution_arn = response["PipelineExecutionArn"]

    print(f"Pipeline execution started.")
    print(f"Execution ARN: {execution_arn}")

    # Emit to GitHub Actions output if running inside Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"execution_arn={execution_arn}\n")

    if not args.wait:
        print("--wait not set; exiting without polling.")
        sys.exit(0)

    final_status = poll_until_complete(sm_client, execution_arn)
    print(f"\nFinal execution status: {final_status}")

    if final_status in SUCCESS_STATUSES:
        print("Pipeline completed successfully.")
        sys.exit(0)
    else:
        print(f"Pipeline did not succeed (status={final_status}).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
