from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml import dsl


import os

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
tenant_id = os.getenv("AZURE_TENANT_ID")
client_id = os.getenv("AZURE_CLIENT_ID")
client_secret = os.getenv("AZURE_CLIENT_SECRET")
credential = ClientSecretCredential(tenant_id, client_id, client_secret)

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

environment_name = 'my-custom-env-2'
environment_version = 1.11
print(f'Using environment {environment_name} version {environment_version}')
compute_target_name = 'adsai-lambda-0'

component_path = "./AxonRooter/src/axonrooter/"
env = ml_client.environments.get(environment_name, environment_version)

train_component = command(
    name="train",
    inputs={
        "data_root": Input(type="uri_folder"),      # just ONE folder
        "pretrained_model": Input(type="uri_file"),
    },
    outputs=dict(model=Output(type="uri_folder", mode="rw_mount")),
    code=component_path,
    command=(
        # pass only the root; script figures out sub-dirs
        "python train.py "
        "--pretrained-model ${{inputs.pretrained_model}} "
        "--data-root        ${{inputs.data_root}} "
        "--model-dir        ${{outputs.model}}"
    ),
    environment=env,
    compute=compute_target_name,
)

train_component = ml_client.create_or_update(train_component.component)

evaluate_component = command(
                        name="evaluate",
                        display_name="Evaluate model",
                        description="Evaluate model with data from a predefined data asset",
                        inputs={
                            "data": Input(type="uri_folder", description="Data asset URI"),
                            "model": Input(type="uri_folder", description="Model URI"),
                        },
                        outputs=dict(
                           accuracy=Output(type="uri_folder", description="Model accuracy output")
                        ),
                        code=component_path,
                        command=(
                            "python evaluate.py "
                            "--use_uri "
                            "--test_data_dir ${{inputs.data}} "
                            "--model_path ${{inputs.model}} "
                            "--accuracy_path ${{outputs.accuracy}}"
                        ),
                        environment=env,
                        compute=compute_target_name,
                    )

evaluate_component = ml_client.create_or_update(evaluate_component.component)

register_component = command(
    name="register",
    description="Register model if it passes accuracy threshold",
    inputs={
        "model": Input(type="custom_model", description="Trained model"),
        "accuracy": Input(type="uri_folder", description="Evaluation results"),
    },
    code=component_path,
    command="python register.py --model ${{inputs.model}} --accuracy ${{inputs.accuracy}}",
    environment=env,
)


register_component = ml_client.create_or_update(register_component.component)

# list all components
components = ml_client.components.list()
print("Components:\n")
for component in components:
    print(component.name, ":", component.version)
print("-------------------")

@dsl.pipeline(
    name='AxonRooter pipeline',
    compute='adsai-lambda-0',
    instance_type="gpu"
)
def train_eval_reg_pipeline(
    pretrained_model: Input,
    train_data_asset_uri: Input,
    test_data_asset_uri: Input,
):
    training_step = train_component(
        pretrained_model=pretrained_model,
        data_root=train_data_asset_uri
    )
    evaluation_step = evaluate_component(
        data=test_data_asset_uri,
        model=training_step.outputs.model
    )
    register_step = register_component(
        model=training_step.outputs.model,
        accuracy=evaluation_step.outputs.accuracy
    )


pipeline_instance = train_eval_reg_pipeline(
    pretrained_model=Input(type="custom_model", path="azureml:michal_unet_v3:2"),
    train_data_asset_uri=Input(type="uri_folder", path="azureml:bigger-segmentation-dataset-only-roots:1"),
    test_data_asset_uri=Input(type="uri_folder", path="azureml:bigger-test-segmentation-dataset-only-roots:1")
)

# Submit the pipeline.
pipeline_run = ml_client.jobs.create_or_update(pipeline_instance)

