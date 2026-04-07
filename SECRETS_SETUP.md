# GitHub Secrets Setup Guide

Before the workflows can run, you need to add the following secrets
to your GitHub repository.

Go to: GitHub Repo → Settings → Secrets and variables → Actions → New repository secret

---

## Required Secrets

| Secret Name           | Description                                              | Example Value |
|-----------------------|----------------------------------------------------------|---------------|
| `AWS_OIDC_ROLE_ARN`   | IAM Role ARN that GitHub Actions assumes via OIDC        | `arn:aws:iam::123456789012:role/GitHubActionsRole` |
| `S3_BUCKET_NAME`      | S3 bucket name for storing data, code and model artifacts| `churnguard-mlops-bucket` |
| `SAGEMAKER_ROLE_ARN`  | IAM Role ARN that SageMaker uses to run jobs             | `arn:aws:iam::123456789012:role/SageMakerExecutionRole` |
| `SNS_TOPIC_ARN`       | SNS Topic ARN for deployment and failure notifications   | `arn:aws:sns:us-east-1:123456789012:churnguard-alerts` |

---

## OIDC Trust Policy for GitHubActionsRole

Add this trust policy to your IAM Role so GitHub Actions can assume it securely:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_GITHUB_USERNAME/customer-churn-prediction:*"
        }
      }
    }
  ]
}
```

---

## Next Steps

1. Create the OIDC provider in AWS IAM (one-time setup per AWS account)
2. Create the GitHubActionsRole with the trust policy above
3. Add all secrets listed above to your GitHub repo
4. Push code to main branch to trigger the first workflow run
