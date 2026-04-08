"""
ChurnGuard CDK Stack
Provisions S3, IAM roles, and SNS for the ChurnGuard MLOps pipeline.
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    RemovalPolicy,
    CfnOutput,
    Duration,
)
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sns as sns
from constructs import Construct


class ChurnGuardStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        account_id = self.account

        # ── S3 Bucket ──────────────────────────────────────────────────────────
        # Stores raw data, processed features, training code, and model artifacts.
        artifacts_bucket = s3.Bucket(
            self,
            "ChurnGuardArtifactsBucket",
            bucket_name=f"churnguard-artifacts-{account_id}",
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            # Change to RemovalPolicy.RETAIN for production
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,  # Required when removal_policy=DESTROY
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="InfrequentAccessAndExpiry",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        )
                    ],
                    expiration=Duration.days(365),
                )
            ],
        )

        # ── SageMaker Execution Role ───────────────────────────────────────────
        # Used by SageMaker jobs (processing, training, evaluation, endpoint).
        sagemaker_role = iam.Role(
            self,
            "ChurnGuardSageMakerRole",
            role_name="ChurnGuardSageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonS3FullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonAthenaFullAccess"
                ),
            ],
        )

        # ── GitHub Actions OIDC Role ───────────────────────────────────────────
        # Allows GitHub Actions to authenticate via OIDC (no long-lived credentials).
        github_oidc_provider = iam.OpenIdConnectProvider.from_open_id_connect_provider_arn(
            self,
            "GitHubOIDCProvider",
            open_id_connect_provider_arn=(
                f"arn:aws:iam::{account_id}:oidc-provider/"
                "token.actions.githubusercontent.com"
            ),
        )

        github_actions_role = iam.Role(
            self,
            "ChurnGuardGitHubActionsRole",
            role_name="ChurnGuardGitHubActionsRole",
            assumed_by=iam.WebIdentityPrincipal(
                github_oidc_provider.open_id_connect_provider_arn,
                conditions={
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": (
                            "repo:dheerajKasha/customer-churn-prediction:*"
                        )
                    },
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                    },
                },
            ),
            inline_policies={
                "ChurnGuardGitHubActionsPolicy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            sid="SageMakerPipelineExecution",
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "sagemaker:StartPipelineExecution",
                                "sagemaker:DescribePipelineExecution",
                                "sagemaker:ListPipelineExecutionSteps",
                                "sagemaker:StopPipelineExecution",
                                "sagemaker:ListPipelineExecutions",
                                "sagemaker:DescribePipeline",
                            ],
                            resources=["*"],
                        ),
                        iam.PolicyStatement(
                            sid="S3ArtifactsAccess",
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:ListBucket",
                            ],
                            resources=[
                                artifacts_bucket.bucket_arn,
                                f"{artifacts_bucket.bucket_arn}/*",
                            ],
                        ),
                    ]
                )
            },
        )

        # ── SNS Topic ─────────────────────────────────────────────────────────
        # Receives alerts for pipeline success, failure, and model drift events.
        alerts_topic = sns.Topic(
            self,
            "ChurnGuardAlertsTopic",
            topic_name="churnguard-alerts",
            display_name="ChurnGuard ML Alerts",
        )

        # ── CloudFormation Outputs ─────────────────────────────────────────────
        # Export values used as GitHub Secrets and in pipeline scripts.
        CfnOutput(
            self,
            "BucketName",
            value=artifacts_bucket.bucket_name,
            description="S3 bucket for ChurnGuard ML artifacts",
            export_name="ChurnGuard-BucketName",
        )

        CfnOutput(
            self,
            "SageMakerRoleArn",
            value=sagemaker_role.role_arn,
            description="IAM Role ARN for SageMaker job execution",
            export_name="ChurnGuard-SageMakerRoleArn",
        )

        CfnOutput(
            self,
            "GitHubActionsRoleArn",
            value=github_actions_role.role_arn,
            description="IAM Role ARN for GitHub Actions OIDC authentication",
            export_name="ChurnGuard-GitHubActionsRoleArn",
        )

        CfnOutput(
            self,
            "SnsTopicArn",
            value=alerts_topic.topic_arn,
            description="SNS Topic ARN for ChurnGuard ML pipeline alerts",
            export_name="ChurnGuard-SnsTopicArn",
        )
