import os
from typing import Optional

import wandb


def find_wandb_run_by_name(model_name: str, path: str):
    api = wandb.Api()
    runs = api.runs(path)

    for run in runs:
        if run.name == model_name:
            print(f"Found run: {run.id} (name: {run.name})")
            return run

    print(f"Run with name '{model_name}' not found in '{path}'.")
    return None


def download_model_for_run(target_run) -> Optional[str]:
    if target_run is None:
        return None

    art = None

    # Prefer a used artifact that looks like a model.
    for a in target_run.used_artifacts():
        if "model" in a.name:
            art = a
            break

    # Fallback: any logged artifact.
    if art is None:
        for a in target_run.logged_artifacts():
            if "model" in a.name:
                art = a
                break
        if art is None:
            for a in target_run.logged_artifacts():
                art = a
                break

    if art is None:
        print("No artifacts found for this run.")
        return None

    download_folder = f"param/models/{target_run.name}"
    os.makedirs(download_folder, exist_ok=True)
    artifact_dir = art.download(root=download_folder)
    print(f"Downloaded artifact '{art.name}' to: {artifact_dir}")
    return artifact_dir
