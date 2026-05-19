from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .utils.convert_pt_utils import (
        create_connection_from_template,
        create_connection_from_template_with_weights,
        create_from_template,
        create_neuron_from_template,
    )
    from .utils.wandb_utils import download_model_for_run, find_wandb_run_by_name
except ImportError:
    # Fallback when running as a standalone script during development
    from utils.convert_pt_utils import (
        create_connection_from_template,
        create_connection_from_template_with_weights,
        create_from_template,
        create_neuron_from_template,
    )
    from utils.wandb_utils import download_model_for_run, find_wandb_run_by_name


SUPPORTED_MODES = {"optimized", "event-driven"}


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _default_template_dir() -> Path:
    return _package_root() / "templates"


def _default_runtime_root() -> Path:
    return _package_root() / "runtimes"


def _runtime_subdir_for_mode(mode: str) -> str:
    if mode == "optimized":
        return "optimized"
    if mode == "event-driven":
        return "event_driven"
    raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {sorted(SUPPORTED_MODES)}")


def _model_display_name(model_path: Path) -> str:
    """
    Return a readable model name for generated headers.

    Rules:
    - if the resolved file is literally named model.pt, use the parent folder name
    - otherwise use the file stem
    """
    if model_path.name == "model.pt" and model_path.parent.name:
        return model_path.parent.name
    return model_path.stem


def _resolve_model_source(
    model_source: str | Path,
    wandb_project: str | None = None,
    models_dir: str | Path | None = None,
) -> Path:
    """
    Resolve a model source into a concrete .pt file path.

    Accepted inputs:
    - direct path to model.pt
    - path to a folder containing model.pt
    - W&B run name, if wandb_project is provided
    """
    source = Path(model_source).expanduser()

    # Case 1: exact local path exists
    if source.exists():
        if source.is_file():
            if source.suffix != ".pt":
                raise ValueError(
                    f"Model file must have '.pt' extension, got: {source}"
                )
            return source.resolve()

        if source.is_dir():
            candidate = source / "model.pt"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Directory '{source}' does not contain 'model.pt'."
                )
            return candidate.resolve()

        raise FileNotFoundError(f"Unsupported filesystem object: {source}")

    # Case 2: W&B run name
    if wandb_project is not None:
        local_models_dir = Path(models_dir or "param/models").expanduser().resolve()
        run_folder = local_models_dir / str(model_source)
        model_file = run_folder / "model.pt"

        if not model_file.exists():
            print(f"Model '{model_source}' not found locally. Downloading from Weights & Biases...")
            target_run = find_wandb_run_by_name(str(model_source), wandb_project)
            if target_run is None:
                raise RuntimeError(
                    f"Could not find W&B run named '{model_source}' in project '{wandb_project}'."
                )
            artifact_dir = download_model_for_run(target_run)
            if artifact_dir is None:
                raise RuntimeError(f"Failed to download artifact for run '{model_source}'.")
            model_file = Path(artifact_dir) / "model.pt"

        if not model_file.exists():
            raise FileNotFoundError(
                f"W&B artifact for '{model_source}' does not contain 'model.pt'."
            )

        return model_file.resolve()

    # Case 3: neither a valid local path nor W&B-enabled run name
    raise FileNotFoundError(
        f"Could not resolve model input '{model_source}'. "
        "Pass a valid .pt file path, a folder containing model.pt, "
        "or provide --wandb-project so it can be treated as a W&B run name."
    )


def _load_state_dict(model_source: str | Path, wandb_project: str | None = None, models_dir: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    model_path = _resolve_model_source(model_source, wandb_project=wandb_project, models_dir=models_dir)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Loaded object from '{model_path}' is not a state_dict dictionary.")
    return state_dict, model_path


def _detect_control_architecture(control_state_dict: dict[str, Any]) -> tuple[bool, str]:
    if "l1.synapse_ff.weight" in control_state_dict:
        print("Control network type: Recurrent (OneLayerRSNN)")
        return True, "l1.synapse_ff.weight"

    if "l1.synapse.weight" in control_state_dict:
        print("Control network type: Feedforward")
        return False, "l1.synapse.weight"

    raise KeyError(
        "Could not detect control architecture. Expected either "
        "'l1.synapse_ff.weight' or 'l1.synapse.weight' in control state_dict."
    )


def _detect_attitude_architecture(attitude_state_dict: dict[str, Any]) -> tuple[bool, str]:
    if "l2.synapse_ff.weight" in attitude_state_dict:
        print("Attitude network type: Recurrent (RSNN)")
        return True, "l2.synapse_ff.weight"

    if "l2.synapse.weight" in attitude_state_dict:
        print("Attitude network type: Feedforward")
        return False, "l2.synapse.weight"

    raise KeyError(
        "Could not detect attitude architecture. Expected either "
        "'l2.synapse_ff.weight' or 'l2.synapse.weight' in attitude state_dict."
    )


def _load_mask(mask_path: str | Path | None, expected_size: int, label: str) -> np.ndarray:
    if mask_path is None:
        return np.ones(expected_size, dtype=np.float32)

    mask = np.loadtxt(mask_path, delimiter=",")
    mask = np.asarray(mask).reshape(-1)

    if mask.size != expected_size:
        raise ValueError(
            f"{label} mask size mismatch. Expected {expected_size} entries, got {mask.size}."
        )

    return mask.astype(np.float32)


def _copy_runtime_files(mode: str, runtime_root: Path, output_dir: Path) -> list[Path]:
    runtime_src = runtime_root / _runtime_subdir_for_mode(mode)
    if not runtime_src.exists():
        raise FileNotFoundError(f"Runtime directory not found: '{runtime_src}'")

    runtime_out = output_dir / "runtime"
    runtime_out.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for path in sorted(runtime_src.iterdir()):
        if path.is_file():
            dst = runtime_out / path.name
            shutil.copy2(path, dst)
            copied.append(dst)

    return copied


def convert_models(
    att_model: str | Path,
    control_model: str | Path,
    mode: str,
    output_dir: str | Path,
    *,
    template_dir: str | Path | None = None,
    runtime_root: str | Path | None = None,
    att_mask_path: str | Path | None = None,
    control_mask_path: str | Path | None = None,
    wandb_project: str | None = None,
    models_dir: str | Path | None = None,
    copy_runtime: bool = True,
) -> dict[str, Any]:

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {sorted(SUPPORTED_MODES)}")

    template_dir = Path(template_dir) if template_dir is not None else _default_template_dir()
    runtime_root = Path(runtime_root) if runtime_root is not None else _default_runtime_root()
    output_dir = Path(output_dir).resolve()
    controller_dir = output_dir / "controller"

    output_dir.mkdir(parents=True, exist_ok=True)
    controller_dir.mkdir(parents=True, exist_ok=True)

    attitude_state_dict, att_model_path = _load_state_dict(
        att_model, wandb_project=wandb_project, models_dir=models_dir
    )
    control_state_dict, control_model_path = _load_state_dict(
        control_model, wandb_project=wandb_project, models_dir=models_dir
    )

    rec_control, control_layer_name = _detect_control_architecture(control_state_dict)
    rec_attitude, attitude_layer_name = _detect_attitude_architecture(attitude_state_dict)

    attitude_enc_layer_name = "l1.synapse.weight"

    if attitude_enc_layer_name not in attitude_state_dict:
        raise KeyError(
            f"Missing required key '{attitude_enc_layer_name}' in attitude state_dict."
        )
    if "l2.neuron.leak_i" not in attitude_state_dict:
        raise KeyError("Missing required key 'l2.neuron.leak_i' in attitude state_dict.")
    if "p_out.synapse.weight" not in control_state_dict:
        raise KeyError("Missing required key 'p_out.synapse.weight' in control state_dict.")

    attitude_mask = _load_mask(
        att_mask_path,
        expected_size=attitude_state_dict[attitude_layer_name].size()[0],
        label="Attitude",
    )
    control_mask = _load_mask(
        control_mask_path,
        expected_size=control_state_dict[control_layer_name].size()[0],
        label="Control",
    )

    n_masked_neurons_attitude = int(np.sum(attitude_mask == 0))
    n_masked_neurons_control = int(np.sum(control_mask == 0))

    attitude_input_size = int(attitude_state_dict[attitude_enc_layer_name].size()[1])
    control_input_size = int(control_state_dict[control_layer_name].size()[1])

    attitude_encoding_size = int(attitude_state_dict[attitude_enc_layer_name].size()[0])
    attitude_hidden_size = int(attitude_state_dict["l2.neuron.leak_i"].size()[0] - n_masked_neurons_attitude)
    control_hidden_size = int(control_state_dict[control_layer_name].size()[0] - n_masked_neurons_control)
    output_size = int(control_state_dict["p_out.synapse.weight"].size()[0])

    print("\n=== Controller Configuration ===")
    print(f"Attitude model:      {att_model_path}")
    print(f"Control model:       {control_model_path}")
    print(f"Attitude input size: {attitude_input_size}")
    print(f"Control input size:  {control_input_size}")
    print(f"Mode:                {mode}")
    print(f"Output directory:    {output_dir}")

    controller_conf_params = {
        "att_name": _model_display_name(att_model_path),
        "control_name": _model_display_name(control_model_path),
        "input_size": attitude_input_size,
        "encoding_size": attitude_encoding_size,
        "hidden_size": attitude_hidden_size,
        "hidden2_size": control_hidden_size,
        "output_size": output_size,
        "type": 1,
    }
    print(f"Controller params: {controller_conf_params}")

    controller_conf_template = template_dir / "test_controller_fwr_conf.templ"
    controller_conf_out = output_dir / "test_controller_fwr_conf.h"

    print("\n=== Generating Header Files ===")
    create_from_template(controller_conf_template, controller_conf_out, controller_conf_params)
    print(f"Generated: {controller_conf_out}")

    # Attitude network export
    create_connection_from_template(
        "inenc",
        attitude_state_dict,
        attitude_enc_layer_name,
        template_dir=template_dir,
        output_dir=controller_dir,
    )
    create_neuron_from_template(
        "enc",
        attitude_state_dict,
        "l1.neuron",
        template_dir=template_dir,
        output_dir=controller_dir,
        sigmoid=True,
    )
    create_connection_from_template(
        "enchid",
        attitude_state_dict,
        attitude_layer_name,
        template_dir=template_dir,
        output_dir=controller_dir,
        mask1=attitude_mask,
    )

    if rec_attitude:
        create_connection_from_template(
            "hidhid",
            attitude_state_dict,
            "l2.synapse_rec.weight",
            template_dir=template_dir,
            output_dir=controller_dir,
            mask1=attitude_mask,
            mask2=attitude_mask,
        )
    else:
        empty_weights = torch.zeros(
            (attitude_hidden_size, attitude_hidden_size), dtype=torch.float32
        )
        create_connection_from_template_with_weights(
            "hidhid",
            empty_weights,
            template_dir=template_dir,
            output_dir=controller_dir,
        )

    create_neuron_from_template(
        "hid",
        attitude_state_dict,
        "l2.neuron",
        template_dir=template_dir,
        output_dir=controller_dir,
        sigmoid=True,
        mask=attitude_mask,
    )
    create_connection_from_template(
        "attout",
        attitude_state_dict,
        "p_out.synapse.weight",
        template_dir=template_dir,
        output_dir=controller_dir,
        mask2=attitude_mask,
    )

    # Control network export
    control_mask_t = torch.as_tensor(control_mask).flatten() != 0
    control_weights = control_state_dict[control_layer_name][control_mask_t, :]

    create_connection_from_template_with_weights(
        "hidhid2",
        control_weights,
        template_dir=template_dir,
        output_dir=controller_dir,
    )

    if rec_control:
        create_connection_from_template(
            "hid2hid2",
            control_state_dict,
            "l1.synapse_rec.weight",
            template_dir=template_dir,
            output_dir=controller_dir,
            mask1=control_mask,
            mask2=control_mask,
        )
    else:
        empty_weights = torch.zeros(
            (control_hidden_size, control_hidden_size), dtype=torch.float32
        )
        create_connection_from_template_with_weights(
            "hid2hid2",
            empty_weights,
            template_dir=template_dir,
            output_dir=controller_dir,
        )

    create_neuron_from_template(
        "hid2",
        control_state_dict,
        "l1.neuron",
        template_dir=template_dir,
        output_dir=controller_dir,
        sigmoid=True,
        mask=control_mask,
    )
    create_connection_from_template(
        "hid2out",
        control_state_dict,
        "p_out.synapse.weight",
        template_dir=template_dir,
        output_dir=controller_dir,
        mask2=control_mask,
    )

    create_from_template(
        template_dir / "test_li_out_file.templ",
        controller_dir / "test_controller_li_out_file.h",
        {"leak": "0.0", "type": "controller"},
    )

    print(f"Generated component headers in: {controller_dir}/")

    copied_runtime_files: list[Path] = []
    if copy_runtime:
        copied_runtime_files = _copy_runtime_files(mode, runtime_root, output_dir)
        print(f"Copied {len(copied_runtime_files)} runtime file(s) to: {output_dir / 'runtime'}")

    return {
        "mode": mode,
        "att_model_path": att_model_path,
        "control_model_path": control_model_path,
        "output_dir": output_dir,
        "controller_dir": controller_dir,
        "config_header": controller_conf_out,
        "copied_runtime_files": copied_runtime_files,
        "params": controller_conf_params,
    }