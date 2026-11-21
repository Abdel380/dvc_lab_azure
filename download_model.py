import shutil
from pathlib import Path
import os
import dagshub
import mlflow

# 1. Connect to DagsHub
TRACKING_URI = "https://dagshub.com/Abdel380/dvc_lab_azure.mlflow"

if os.getenv("MLFLOW_TRACKING_PASSWORD"):
    # Non-interactive authentication (Docker / GitHub Actions)
    mlflow.set_tracking_uri(TRACKING_URI)
else:
    # Interactive authentication (local development)
    dagshub.init(
        repo_owner="Abdel380",
        repo_name="dvc_lab_azure",
        mlflow=True
    )

# 2. Find the MLflow experiment
experiment = mlflow.get_experiment_by_name(
    "RandomForest_V1.0.1"
)

if experiment is None:
    raise RuntimeError("Experiment not found")

# 3. Retrieve all logged models
models = mlflow.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    output_format="list"
)

# 4. Keep only successful models
successful_models = [
    model for model in models
    if model.name == "random_forest_model"
    and model.status == "READY"
    and model.source_run_id
    and mlflow.get_run(model.source_run_id).info.status == "FINISHED"
]

if not successful_models:
    raise RuntimeError("No successful model found")

# 5. Select the most recently created model
latest_model = max(
    successful_models,
    key=lambda model: model.creation_timestamp
)

model_uri = f"models:/{latest_model.model_id}"

print("Latest model:", latest_model.model_id)
print("Downloading model...")

# 6. Download from DagsHub
downloaded_path = mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri
)

# 7. Save to local project directory
destination = Path(__file__).resolve().parent / "model"

if destination.exists():
    shutil.rmtree(destination)

shutil.copytree(downloaded_path, destination)

print("Model downloaded successfully!")
print("Saved to:", destination)