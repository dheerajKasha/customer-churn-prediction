"""
ChurnGuard — SageMaker Pipeline DAG
Defines the full ML pipeline: preprocessing → training → HPO → evaluation →
conditional model registration in SageMaker Model Registry.
"""

import os
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.tuning_step import TuningStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter,
)
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

PIPELINE_NAME = "ChurnGuardPipeline"
MODEL_GROUP_NAME = "ChurnGuardModels"
REGION = "us-east-1"


def get_pipeline(role: str, bucket: str, region: str = REGION) -> Pipeline:
    """
    Build and return the ChurnGuard SageMaker Pipeline.

    Parameters
    ----------
    role   : IAM role ARN for SageMaker to assume when running pipeline steps.
    bucket : S3 bucket used for data, code artefacts, and model outputs.
    region : AWS region (default us-east-1).
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=bucket,
    )

    # ------------------------------------------------------------------
    # Pipeline parameters
    # ------------------------------------------------------------------
    input_data_url = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{bucket}/data/raw/telco_churn.csv",
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.large",
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )
    auc_threshold = ParameterFloat(
        name="AUCThreshold",
        default_value=0.75,
    )

    # ------------------------------------------------------------------
    # Step 1 — Preprocessing
    # ------------------------------------------------------------------
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="churn-preprocessing",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    preprocessing_step = ProcessingStep(
        name="ChurnPreprocessing",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data_url,
                destination="/opt/ml/processing/input",
                input_name="raw-data",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/pipeline/preprocessed/train",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/pipeline/preprocessed/test",
            ),
        ],
        code="src/preprocessing.py",
    )

    # ------------------------------------------------------------------
    # Step 2 — Baseline Training (single run, feeds HPO warm start later)
    # ------------------------------------------------------------------
    xgboost_image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    xgb_estimator = Estimator(
        image_uri=xgboost_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=f"s3://{bucket}/pipeline/models",
        base_job_name="churn-training",
        role=role,
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "max_depth": 6,
            "eta": 0.1,
            "num_round": 100,
            "objective": "binary:logistic",
            "eval_metric": "auc",
        },
    )

    training_step = TrainingStep(
        name="ChurnTraining",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ------------------------------------------------------------------
    # Step 3 — HPO (Hyperparameter Optimisation)
    # ------------------------------------------------------------------
    hpo_estimator = Estimator(
        image_uri=xgboost_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=f"s3://{bucket}/pipeline/hpo-models",
        base_job_name="churn-hpo",
        role=role,
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "objective": "binary:logistic",
            "eval_metric": "auc",
        },
    )

    tuner = HyperparameterTuner(
        estimator=hpo_estimator,
        objective_metric_name="validation:auc",
        hyperparameter_ranges={
            "max_depth": IntegerParameter(3, 10),
            "eta": ContinuousParameter(0.01, 0.3),
            "num_round": IntegerParameter(50, 300),
        },
        metric_definitions=[
            {"Name": "validation:auc", "Regex": ".*\\[\\d+\\].*validation-auc:(\\S+).*"}
        ],
        max_jobs=10,
        max_parallel_jobs=2,
        objective_type="Maximize",
        base_tuning_job_name="churn-hpo",
    )

    hpo_step = TuningStep(
        name="ChurnHPO",
        tuner=tuner,
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ------------------------------------------------------------------
    # Step 4 — Evaluation
    # ------------------------------------------------------------------
    eval_processor = ScriptProcessor(
        image_uri=image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            py_version="py3",
            instance_type=processing_instance_type,
        ),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="churn-evaluation",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluation_step = ProcessingStep(
        name="ChurnEvaluation",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=hpo_step.get_top_model_s3_uri(
                    top_k=0,
                    s3_bucket=bucket,
                    prefix="pipeline/hpo-models",
                ),
                destination="/opt/ml/processing/model",
                input_name="model",
            ),
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
                input_name="test-data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/pipeline/evaluation",
            )
        ],
        code="src/evaluate.py",
        property_files=[evaluation_report],
    )

    # ------------------------------------------------------------------
    # Step 5 — Conditional model registration (AUC gate)
    # ------------------------------------------------------------------
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0][
                    "S3Output"
                ]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    best_model = Model(
        image_uri=xgboost_image_uri,
        model_data=hpo_step.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=bucket,
            prefix="pipeline/hpo-models",
        ),
        sagemaker_session=sagemaker_session,
        role=role,
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=best_model.register(
            content_types=["text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=MODEL_GROUP_NAME,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
        ),
    )

    condition_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="classification_metrics.auc.value",
        ),
        right=auc_threshold,
    )

    check_auc_step = ConditionStep(
        name="CheckAUC",
        conditions=[condition_auc],
        if_steps=[register_step],
        else_steps=[],
    )

    # ------------------------------------------------------------------
    # Assemble pipeline
    # ------------------------------------------------------------------
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            input_data_url,
            processing_instance_type,
            training_instance_type,
            model_approval_status,
            auc_threshold,
        ],
        steps=[
            preprocessing_step,
            training_step,
            hpo_step,
            evaluation_step,
            check_auc_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


# ------------------------------------------------------------------
# Entry point — upsert pipeline definition to SageMaker
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upsert ChurnGuard SageMaker Pipeline")
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("SAGEMAKER_ROLE_ARN"),
        required=not os.environ.get("SAGEMAKER_ROLE_ARN"),
        help="IAM role ARN for SageMaker (or set SAGEMAKER_ROLE_ARN env var)",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET_NAME"),
        required=not os.environ.get("S3_BUCKET_NAME"),
        help="S3 bucket name (or set S3_BUCKET_NAME env var)",
    )
    parser.add_argument("--region", default=REGION, help="AWS region")
    args = parser.parse_args()

    pipeline = get_pipeline(role=args.role_arn, bucket=args.bucket, region=args.region)
    upsert_response = pipeline.upsert(role_arn=args.role_arn)
    print(f"Pipeline upserted: {upsert_response['PipelineArn']}")
