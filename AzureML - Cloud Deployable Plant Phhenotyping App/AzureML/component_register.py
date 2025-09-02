from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id = 
    resource_group = 
    workspace_name = 
)

# Define the component
train_component = command(
    name="train_component",
    display_name="Train Component",
    version="1",
    inputs={
        "train_data": Input(type="uri_folder"),
        "learning_rate": Input(type="number", default=0.001)
    },
    outputs={
        "model_output": Output(type="uri_folder")
    },
    code=r"C:\Users\emilp\Documents\GitHub\2024-25d-fai2-adsai-group-cv9\AxonRooter\src\axonrooter",
    command=(
        "python train.py "
        "--train_data ${{inputs.train_data}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--model_output ${{outputs.model_output}}"
    ),
    environment="azureml:my-custom-env-2:1.11"
)

# Register the component
ml_client.components.create_or_update(train_component)
