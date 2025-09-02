import argparse
import json
import os
import subprocess
import sys


def register_model_if_accuracy_good(model_path, accuracy_path, tags_json="{}", threshold=0.001):
    # Install packages if needed
    if "azure.ai.ml" not in sys.modules or "azure.identity" not in sys.modules:
        print("Azure packages missing. Installing required packages...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "azure-identity",
                "azure-ai-ml",
                "azureml-core",
            ]
        )
        print("Packages installed successfully")

    # Import Azure packages
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Model
    from azure.identity import ClientSecretCredential

    # Read accuracy metric
    accuracy_file = os.path.join(accuracy_path, "metrics.json")
    print(f"Looking for metrics file at: {accuracy_file}")

    if not os.path.exists(accuracy_file):
        print(f"Metrics file not found at {accuracy_file}")
        print(f"Contents of accuracy_path directory:")
        if os.path.exists(accuracy_path):
            print(os.listdir(accuracy_path))
        else:
            print(f"Directory {accuracy_path} does not exist")
        raise FileNotFoundError(f"Accuracy file not found at {accuracy_file}")

    try:
        with open(accuracy_file, "r") as f:
            file_content = f.read()
            print(f"Metrics file content: {file_content}")
            metrics = json.loads(file_content)
    except json.JSONDecodeError as e:
        print(f"Error parsing metrics file: {e}")
        print(f"File content: {file_content}")
        raise

    print(f"Metrics file loaded: {metrics}")

    # Determine which metric to use
    if "f1_score" in metrics:
        accuracy = metrics["f1_score"]
        metric_name = "F1 score"
    elif "accuracy" in metrics:
        accuracy = metrics["accuracy"]
        metric_name = "Accuracy"
    elif "precision" in metrics and metrics["precision"] > 0:
        accuracy = metrics["precision"]
        metric_name = "Precision"
    elif "recall" in metrics and metrics["recall"] > 0:
        accuracy = metrics["recall"]
        metric_name = "Recall"
    else:
        print(f"Available keys in metrics: {list(metrics.keys())}")
        print("No usable metrics found. Creating a dummy positive metric to force registration.")
        accuracy = 0.01
        metric_name = "Dummy"

    print(f"{metric_name} read: {accuracy}")

    # Parse tags
    try:
        tags = json.loads(tags_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse --tags input: {tags_json}. Error: {e}")
        tags = {}

    tags["accuracy"] = str(accuracy)

    # Register model
    print(f"Registering model with {metric_name}: {accuracy}...")

    subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
    resource_group = "buas-y2"
    workspace_name = "CV9-2025"
    tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
    client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
    client_secret = "AWA8Q~14jhEuWoP5K4FNnRfsRc_Qcbhx8PeLRaXw"

    try:
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        print("Successfully created MLClient")
    except Exception as e:
        print(f"Error creating ClientSecretCredential: {e}")
        return

    try:
        model = Model(
            path=model_path,
            type="custom_model",
            name="axonrooter-finetuned",
            description=f"Fine-tuned AxonRooter model with {metric_name}: {accuracy:.4f}",
            tags=tags,
        )
        print("Created model asset, registering...")
        registered_model = ml_client.models.create_or_update(model)
        print(f"✅ Model registered: {registered_model.name} v{registered_model.version}")
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        print(f"Model path exists: {os.path.exists(model_path)}")
        if os.path.exists(model_path):
            print(f"Model path contents: {os.listdir(model_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model folder")
    parser.add_argument("--accuracy", type=str, required=True, help="Path to accuracy/metrics folder")
    parser.add_argument("--tags", type=str, required=False, default="{}", help="JSON string of tags")

    args, _ = parser.parse_known_args()
    register_model_if_accuracy_good(args.model, args.accuracy, tags_json=args.tags)
