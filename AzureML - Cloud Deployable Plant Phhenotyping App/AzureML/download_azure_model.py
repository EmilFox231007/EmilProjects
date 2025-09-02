from azureml.core import Workspace, Model

# Load your Azure ML workspace
ws = Workspace.from_config()  # Or use Workspace.get() with parameters

# List all registered models
for model in Model.list(ws):
    print(model.name, model.version)

# Download a specific model by name (and version optionally)
model = Model(ws, name="axonrooter-finetuned", version=10)  # Or use version=2 if needed
model.download(target_dir=r"C:\Users\emilp\Downloads", exist_ok=True)
