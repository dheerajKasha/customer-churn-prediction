# ChurnGuard — Claude Code Context

## Project Overview
**Name:** ChurnGuard
**Repo Path:** C:\Users\dheer\Documents\repos\customer-churn-prediction
**AWS Region:** us-east-1

**One Liner:**
A production-grade MLOps project that automatically detects telecom customer churn risk, retrains itself when new data arrives, and serves live predictions through a REST API — all triggered from a GitHub push with zero manual AWS intervention.

---

## Tech Stack
Python 3.11, AWS SageMaker, SageMaker Pipelines, SageMaker Feature Store, SageMaker Model Monitor, AWS S3, AWS Glue, Amazon Athena, AWS Lambda, Amazon API Gateway, Amazon CloudWatch, Amazon SNS, GitHub Actions, AWS OIDC, AWS IAM, AWS CDK, XGBoost, scikit-learn, boto3, pandas, numpy

---

## Architecture

```
IBM Telco CSV (Kaggle)
        ↓
       S3  ←─────────────────────────────────────────┐
        ↓                                             │
GitHub Push / Daily CRON                             SNS Alert
        ↓                                             │
GitHub Actions (OIDC → IAM Role)                     │
        ↓                                             │
SageMaker Pipeline DAG                               │
  ├── Processing Job (Glue + Feature Store)          │
  ├── Training Job (XGBoost)                         │
  ├── HPO (Auto Model Tuning)                        │
  ├── Evaluation + Experiment Tracking               │
  └── Model Registry (Approval Gate)                 │
        ↓                                             │
SageMaker Endpoint                                   │
        ↓                                             │
Lambda + API Gateway                                 │
        ↓                                             │
Model Monitor ──── Drift Detected ───────────────────┘
        ↓
CloudWatch Dashboard
```

---

## Dataset
- **Name:** IBM Telco Customer Churn
- **Source:** Kaggle (free, open license)
- **Size:** ~7,000 rows
- **Target column:** `Churn` (Yes/No)
- **Download:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Local path once downloaded:** `data/telco_churn.csv`

---

## Folder Structure
```
customer-churn-prediction/
├── .github/
│   └── workflows/
│       ├── train.yml                  ✅ DONE - triggers on push to src/ or pipeline/
│       ├── deploy.yml                 ✅ DONE - manual trigger with approval gate
│       └── scheduled_retrain.yml     ✅ DONE - daily 6AM UTC cron retraining
├── pipeline/
│   ├── sagemaker_pipeline.py          ⬜ TODO - defines the SageMaker Pipeline DAG
│   ├── run_pipeline.py                ⬜ TODO - triggers pipeline execution via boto3
│   ├── approve_model.py               ⬜ TODO - approves model in Model Registry
│   ├── deploy_model.py                ⬜ TODO - deploys approved model to endpoint
│   └── check_new_data.py             ⬜ TODO - checks S3 for new data, sets output flag
├── src/
│   ├── preprocessing.py              ⬜ TODO - feature engineering, data cleaning
│   ├── train.py                      ⬜ TODO - XGBoost training script
│   └── evaluate.py                   ⬜ TODO - model evaluation, metrics output
├── infra/
│   └── cdk_stack.py                  ⬜ TODO - AWS CDK stack (S3, IAM, SNS, endpoints)
├── tests/
│   └── test_preprocessing.py         ⬜ TODO - unit tests for preprocessing
├── notebooks/
│   └── exploration.ipynb             ⬜ TODO - EDA notebook for Telco dataset
├── data/
│   └── telco_churn.csv               ⬜ TODO - download from Kaggle and place here
├── CLAUDE.md                         ✅ DONE - this file
├── SECRETS_SETUP.md                  ✅ DONE - GitHub secrets and OIDC setup guide
└── requirements.txt                  ✅ DONE - all Python dependencies
```

---

## GitHub Secrets Required
| Secret Name          | Description                                      |
|----------------------|--------------------------------------------------|
| `AWS_OIDC_ROLE_ARN`  | IAM Role ARN assumed by GitHub Actions via OIDC  |
| `S3_BUCKET_NAME`     | S3 bucket for data, code, and model artifacts    |
| `SAGEMAKER_ROLE_ARN` | IAM Role ARN used by SageMaker to run jobs       |
| `SNS_TOPIC_ARN`      | SNS Topic ARN for alerts and notifications       |

---

## What Is Done
- [x] Project folder structure created
- [x] GitHub Actions workflow — train.yml (code push trigger)
- [x] GitHub Actions workflow — deploy.yml (manual approval + deploy)
- [x] GitHub Actions workflow — scheduled_retrain.yml (daily cron)
- [x] requirements.txt
- [x] SECRETS_SETUP.md with OIDC trust policy template
- [x] CLAUDE.md (this file)

## What Is Next (Build in this order)
1. **infra/cdk_stack.py** — provision S3, IAM roles, SNS topic via AWS CDK
2. **pipeline/sagemaker_pipeline.py** — define the full SageMaker Pipeline DAG
3. **pipeline/run_pipeline.py** — boto3 script to trigger pipeline execution
4. **pipeline/check_new_data.py** — S3 data check for scheduled retraining
5. **pipeline/approve_model.py** — approve model in SageMaker Model Registry
6. **pipeline/deploy_model.py** — deploy approved model to SageMaker endpoint
7. **src/preprocessing.py** — data cleaning and feature engineering
8. **src/train.py** — XGBoost training script for SageMaker
9. **src/evaluate.py** — model evaluation and metrics
10. **tests/test_preprocessing.py** — unit tests
11. **notebooks/exploration.ipynb** — EDA on Telco dataset

---

## Key Decisions Made
- **No hardcoded AWS credentials** — GitHub Actions uses OIDC to assume IAM role securely
- **XGBoost** chosen as the model — explainable, fast, well supported in SageMaker
- **CDK in Python** — consistent with the rest of the project language
- **Manual approval gate** in deploy.yml before any production deployment
- **SNS notifications** on both success and failure of deployments
- **Daily cron** checks for new data before triggering retraining — avoids unnecessary runs

---

## How to Continue in Claude Code
Open terminal in the repo root and say:
> "Read CLAUDE.md and continue building ChurnGuard — start with infra/cdk_stack.py"
