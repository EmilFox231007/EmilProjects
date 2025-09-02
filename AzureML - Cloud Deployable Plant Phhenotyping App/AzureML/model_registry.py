# register_model.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model               # <-- import Model entity
from azure.identity import DefaultAzureCredential   
from azure.ai.ml.constants import AssetTypes # or InteractiveBrowserCredential

# ------------------------------------------------------------------
# 1.  Workspace details  (all strings!)
subscription_id   = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group    = "buas-y2"
workspace_name    = "CV9-2025"

# ------------------------------------------------------------------
# 2.  Create MLClient  (assumes `az login` or env-vars for DefaultAzureCredential)
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name
)

# ------------------------------------------------------------------
# 3.  Describe the model asset
model_asset = Model(
    path="C:\\Users\\emilp\\Documents\\GitHub\\2024-25d-fai2-adsai-group-cv9 - Copymain\\AxonRooter\\src\\axonrooter\\models\\michal_232079_unet_model_v3_256px.h5",  # local file -> uploaded
    name="michal_unet_v3",                                 # registry name
    version="2",                 
    description="UNet model for AxonRooter retraining",                          # set/override as needed
    type=AssetTypes.CUSTOM_MODEL,                                # or AssetTypes.CUSTOM_MODEL
)

# ------------------------------------------------------------------
# 4.  Register (creates or updates)
registered = ml_client.models.create_or_update(model_asset)
print(f"Registered: {registered.name}:{registered.version}")
