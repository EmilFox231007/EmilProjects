'''from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime, timedelta
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml import Input, dsl, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import KubernetesOnlineDeployment, CodeConfiguration, ResourceSettings, ResourceRequirementsSettings

# === Constants ===
DATA_ASSET_NAME = "axonrooter_student_corrections_Endijs_20250626_040517"
MODEL_NAME = "axonrooter-finetuned"
ENV_NAME = "my-custom-env-2"
ENV_VERSION = "1.11"
ENDPOINT_NAME = "cv9-endpoint-06251752354033"
CODE_PATH = r"C:\Users\emilp\Documents\GitHub\2024-25d-fai2-adsai-group-cv9\AxonRooter\src\axonrooter"
SCORING_SCRIPT = "scoring.py"
ACCURACY_THRESHOLD = 0.80  # Minimum required to promote to green

code_configuration = CodeConfiguration(
    code=CODE_PATH,
    scoring_script=SCORING_SCRIPT
)


# === DAG Default Args ===
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1)
}

def get_ml_client():
    conn = BaseHook.get_connection("azure_ml_conn")
    extra = conn.extra_dejson
    credential = ClientSecretCredential(
        tenant_id=extra["tenant_id"],
        client_id=extra["client_id"],
        client_secret=extra["client_secret"]
    )
    return MLClient(
        credential=credential,
        subscription_id=extra["subscription_id"],
        resource_group_name=extra["resource_group"],
        workspace_name=extra["workspace_name"]
    )

def run_pipeline():
    ml_client = get_ml_client()
    dataset = ml_client.data.get(name=DATA_ASSET_NAME, label="latest")

    train_component = ml_client.components.get("train2", version="5")
    eval_component = ml_client.components.get("evaluate2", version="2")
    register_component = ml_client.components.get("register2", version="2")
    pretrained_model = ml_client.models.get(name=MODEL_NAME, version="8")

    @dsl.pipeline(compute="adsai-lambda-2", description="Weekly retraining pipeline")
    def retraining_pipeline():
        train = train_component(
            data_root=Input(type=AssetTypes.URI_FOLDER, path=dataset.id),
            pretrained_model=Input(type=AssetTypes.URI_FILE, path=pretrained_model.path),
        )
        eval = eval_component(
            model=train.outputs.model,
            data=Input(type=AssetTypes.URI_FOLDER, path="azureml:bigger-test-segmentation-dataset-only-roots:1"),
        )
        register = register_component(
            model=train.outputs.model,
            accuracy=eval.outputs.accuracy,
            tags="{'experiment':'weekly-retrain'}"  # Example, optional
        )

    pipeline_job = retraining_pipeline()
    submitted_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="axonrooter_weekly_retrain")
    ml_client.jobs.stream(submitted_job.name)
    print("🚀 Submitted retraining pipeline job and waited for completion.")

def promote_model_to_green():
    ml_client = get_ml_client()

    # Get latest registered model
    models = ml_client.models.list(name=MODEL_NAME)
    latest_model = max(models, key=lambda m: int(m.version))

    # Get accuracy from tags (you must ensure the register component sets this)
    accuracy_str = latest_model.tags.get("accuracy")
    if not accuracy_str:
        print("⚠️ No accuracy tag found. Skipping deployment.")
        return

    accuracy = float(accuracy_str)
    print(f"✅ Latest model version: {latest_model.version} with accuracy {accuracy}")

    if accuracy < ACCURACY_THRESHOLD:
        print("❌ Accuracy too low. Not promoting to green.")
        return

    # Update green deployment
    env = ml_client.environments.get(name=ENV_NAME, version=ENV_VERSION)
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
    print("✅ Promoted model to green deployment.")

# === DAG Definition ===
with DAG(
    dag_id="axonrooter_weekly_retraining",
    default_args=default_args,
    schedule_interval="@weekly",
    catchup=False,
    tags=["axonrooter", "azureml", "retraining"]
) as dag:

    run_pipeline_task = PythonOperator(
        task_id="run_retraining_pipeline",
        python_callable=run_pipeline,
    )

    promote_model_task = PythonOperator(
        task_id="promote_model_to_green",
        python_callable=promote_model_to_green,
    )

    run_pipeline_task >> promote_model_task'''
