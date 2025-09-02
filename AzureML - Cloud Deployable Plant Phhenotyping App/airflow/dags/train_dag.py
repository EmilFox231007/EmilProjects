'''from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Connection
from airflow.hooks.base import BaseHook
from azure.ai.ml import MLClient, Input, Output
from azure.identity import ClientSecretCredential
from datetime import datetime


def submit_job():
    conn = BaseHook.get_connection("azure_ml_conn")
    extra = conn.extra_dejson

    credential = ClientSecretCredential(
        tenant_id=extra["tenant_id"],
        client_id=extra["client_id"],
        client_secret=extra["client_secret"]
    )

    print("Resource group passed to MLClient:", extra.get("resource_group"))

    ml_client = MLClient(
        credential=credential,
        subscription_id=extra["subscription_id"],
        resource_group=extra["resource_group"],
        workspace_name=extra["workspace_name"]
    )

    component = ml_client.components.get("train")

    job = component(
        inputs={
            "data_root": Input(path="azureml://datastores/workspaceblobstore/paths/data/", mode="ro_mount"),
            "pretrained_model": Input(path="azureml://datastores/workspaceblobstore/paths/models/michal_v3.h5", mode="ro_mount")
        },
        outputs={
            "model_output": Output(mode="upload")
        }
    )

    ml_client.jobs.create_or_update(job)


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "depends_on_past": False,
}

with DAG(
    dag_id="axonrooter_train_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["azureml", "axonrooter"]
) as dag:
    train_model_task = PythonOperator(
        task_id="submit_azureml_training_job",
        python_callable=submit_job
    )
'''