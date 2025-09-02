from azure.ai.ml import Input, MLClient, command, Output
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes

credential = InteractiveBrowserCredential()

subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "CV9-2025"

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

env = ml_client.environments.get("my-custom-env-2", version="1.11")
compute_target = "adsai-lambda-0"
vm_size = "gpu"  # or remove this line if using Kubernetes and letting Azure choose

data_asset = ml_client.data.get("bigger-segmentation-dataset", version="1")

pretrained_asset = ml_client.models.get("michal_unet_v3", version="2")

job = command(
    code="./AxonRooter/src/axonrooter",
    command=(
        "python train.py "
        "--pretrained-model ${{inputs.pretrained}} "
        "--data-root        ${{inputs.data_root}} "
        "--model-dir        ${{outputs.model}}"
    ),
    inputs={
        "pretrained": Input(
            path=pretrained_asset.id,  
            type=AssetTypes.CUSTOM_MODEL,
            mode=InputOutputModes.RO_MOUNT
        ),
        "data_root": Input(
            path=data_asset.id,
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RO_MOUNT
        ),
    },
    outputs={
        "model": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")
    },
    environment=env,
    compute=compute_target,
    instance_type=vm_size,
    display_name="axonrooter_finetune_job",
)

returned_job = ml_client.jobs.create_or_update(job)
