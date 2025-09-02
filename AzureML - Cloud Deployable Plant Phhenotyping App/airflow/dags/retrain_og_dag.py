from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime, timedelta
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml import Input, dsl, Output
from azure.ai.ml.constants import AssetTypes

# === Constants === hello this is me 
DATA_ASSET_NAME = "bigger-segmentation-dataset-only-roots"

# === DAG Default Args ===
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1)
}

def run_pipeline():
    # Load Azure ML connection credentials
    conn = BaseHook.get_connection("azure_ml_conn")
    extra = conn.extra_dejson

    credential = ClientSecretCredential(
        tenant_id=extra["tenant_id"],
        client_id=extra["client_id"],
        client_secret=extra["client_secret"]
    )

    ml_client = MLClient(
        credential=credential,
        subscription_id=extra["subscription_id"],
        resource_group_name=extra["resource_group"],
        workspace_name=extra["workspace_name"]
    )

    # Get the latest version of the user-uploaded dataset (already registered by the app)
    dataset = ml_client.data.get(name=DATA_ASSET_NAME, label="latest")

    # Load components
    train_component = ml_client.components.get("train2", version="5")
    eval_component = ml_client.components.get("evaluate2", version="2")
    register_component = ml_client.components.get("register2", version="2 ")

    pretrained_model = ml_client.models.get(name="axonrooter-finetuned", version="8")

    @dsl.pipeline(compute="adsai-lambda-2", description="Weekly retraining pipeline")
    def retraining_pipeline():
        train = train_component(
            data_root=Input(type=AssetTypes.URI_FOLDER, path=dataset.id),
            pretrained_model=Input(type=AssetTypes.URI_FILE, path=pretrained_model.path),
        )

        eval = eval_component(
            model=train.outputs.model,  # <-- use correct output name
            data=Input(type=AssetTypes.URI_FOLDER, path="azureml:bigger-test-segmentation-dataset-only-roots:1"),
            #accuracy=Output(type=AssetTypes.URI_FOLDER),  # <-- match output name if needed
        )

        register = register_component(
            model=train.outputs.model,
            accuracy=eval.outputs.accuracy,
            tags="{'experiment':'weekly-retrain'}"
        )

    job = retraining_pipeline()
    ml_client.jobs.create_or_update(job, experiment_name="axonrooter_weekly_retrain")
    print("🚀 Submitted retraining pipeline job.")


# === DAG Definition ===
with DAG(
    dag_id="axonrooter_weekly_retraining",
    default_args=default_args,
    schedule_interval="@weekly",  # Every 10 minutes
    catchup=False,
    tags=["axonrooter", "azureml", "retraining"]
) as dag:

    step_run_pipeline = PythonOperator(
        task_id="run_retraining_pipeline",
        python_callable=run_pipeline,
    )

    step_run_pipeline