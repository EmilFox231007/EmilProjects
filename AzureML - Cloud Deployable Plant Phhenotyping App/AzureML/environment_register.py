'''from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext


# Create environment object
env = Environment(
    name="cv_env_azure",
    description="Custom env with image build forced to Kubernetes cluster",
    image="endijskirsteins234924/axonrooter-image:latest-test",
    conda_file="environment.yml",
)

# Register
ml_client.environments.create_or_update(env)'''



from azure.ai.ml.entities import Environment
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
 
credential = InteractiveBrowserCredential()
ml_client = MLClient.from_config(credential=credential)
 
# Create or update an environment using a Docker image
env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    name="my-custom-env-2",
    version= "1.11",
    description="My custom environment using Docker image",
    conda_file="environment.yml",
)
 
ml_client.environments.create_or_update(env)
