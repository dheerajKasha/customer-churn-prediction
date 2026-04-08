import aws_cdk as cdk
from cdk_stack import ChurnGuardStack

app = cdk.App()
ChurnGuardStack(app, "ChurnGuardStack", env=cdk.Environment(region="us-east-1"))
app.synth()
