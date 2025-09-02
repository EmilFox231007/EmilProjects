import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import KubernetesOnlineDeployment, CodeConfiguration, ResourceSettings, ResourceRequirementsSettings

# === Fill in your AzureML details ===
TENANT_ID = "0a33589b-0036-4fe8-a829-3ed0926af886"
CLIENT_ID = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
CLIENT_SECRET = "AWA8Q~14jhEuWoP5K4FNnRfsRc_Qcbhx8PeLRaXw"
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "CV9-2025"

MODEL_NAME = "axonrooter-finetuned"
ENV_NAME = "my-custom-env-2"
ENV_VERSION = "1.11"
ENDPOINT_NAME = "cv9-endpoint-06251752354033"
CODE_PATH = "AxonRooter/src/axonrooter"
SCORING_SCRIPT = "scoring.py"
ACCURACY_THRESHOLD = 0.80

# Authenticate
credential = ClientSecretCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# Get latest registered model
models = list(ml_client.models.list(name=MODEL_NAME))
if not models:
    print("No models found.")
    exit(1)
latest_model = max(models, key=lambda m: int(m.version))

accuracy_str = latest_model.tags.get("accuracy")
if not accuracy_str:
    print("No accuracy tag found. Skipping deployment.")
    exit(1)
accuracy = float(accuracy_str)
print(f"Latest model version: {latest_model.version} with accuracy {accuracy}")

if accuracy < ACCURACY_THRESHOLD:
    print("Accuracy too low. Not promoting to green.")
    exit(0)

env = ml_client.environments.get(name=ENV_NAME, version=ENV_VERSION)
code_configuration = CodeConfiguration(
    code=CODE_PATH,
    scoring_script=SCORING_SCRIPT
)
green_deployment = KubernetesOnlineDeployment(
    name="green",
    endpoint_name=ENDPOINT_NAME,
    model=latest_model,
    environment=env,
    code_configuration=code_configuration,
    instance_count=1,
    resources=ResourceRequirementsSettings(
        requests=ResourceSettings(cpu="8", memory="0.5Gi")
    )
)
ml_client.online_deployments.begin_create_or_update(green_deployment).wait()
print("Promoted model to green deployment.")