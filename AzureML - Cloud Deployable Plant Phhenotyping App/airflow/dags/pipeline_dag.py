'''from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from datetime import datetime
from azure.ai.ml import MLClient, dsl, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential


def submit_pipeline():
    # Load connection from Airflow
    conn = BaseHook.get_connection("azure_ml_conn")
    extra = conn.extra_dejson

    # Use credentials from connection
    credential = ClientSecretCredential(
        tenant_id=extra["tenant_id"],
        client_id=extra["client_id"],
        client_secret=extra["client_secret"]
    )

    ml_client = MLClient(
        credential=credential,
        subscription_id=extra["subscription_id"],
        resource_group_name=extra["resource_group"],
        workspace_name=extra["workspace_name"],
    )

    # Load components (ensure these are registered!)
    train_component = ml_client.components.get("train", version="1")
    eval_component = ml_client.components.get("evaluate", version="1")
    register_component = ml_client.components.get("register", version="1")

    # Define pipeline
    @dsl.pipeline(compute="cpu-cluster", description="AxonRooter full pipeline")
    def axonrooter_pipeline():
        # Step 1: Train
        train = train_component(
            data_root=Input(type=AssetTypes.URI_FOLDER, path="azureml:bigger-segmentation-dataset:2"),
            pretrained_model=Input(type=AssetTypes.URI_FOLDER, path="azureml:axonrooter-finetuned:6"),
            model_dir=Output(type=AssetTypes.URI_FOLDER),
        )

        # Step 2: Evaluate
        eval = eval_component(
            model_path=train.outputs.model_dir,
            test_data_dir=Input(type=AssetTypes.URI_FOLDER, path="azureml:bigger-test-segmentation-dataset:2"),
            accuracy_path=Output(type=AssetTypes.URI_FOLDER),
        )

        # Step 3: Register
        register = register_component(
            model=train.outputs.model_dir,
            accuracy=eval.outputs.accuracy_path,
        )

    # Submit pipeline job
    pipeline_job = axonrooter_pipeline()
    ml_client.jobs.create_or_update(pipeline_job, experiment_name="axonrooter_pipeline")


# Define Airflow DAG
default_args = {"owner": "airflow", "start_date": datetime(2024, 1, 1)}

with DAG(
    dag_id="axonrooter_pipeline_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["azureml", "axonrooter"],
) as dag:
    run_pipeline_task = PythonOperator(
        task_id="run_pipeline",
        python_callable=submit_pipeline,
    )
'''