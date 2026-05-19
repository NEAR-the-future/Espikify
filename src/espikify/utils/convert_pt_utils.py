from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

import torch


def _ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_bool_mask(mask: Any | None, expected_size: int, name: str) -> torch.Tensor | None:
    if mask is None:
        return None

    mask_t = torch.as_tensor(mask).flatten()
    if mask_t.numel() != expected_size:
        raise ValueError(
            f"{name} mask size mismatch. Expected {expected_size}, got {mask_t.numel()}."
        )

    return mask_t != 0


def _format_c_float(value: float) -> str:
    return f"{float(value):.6f}f"


def create_from_template(
    template_filename: str | Path,
    output_filename: str | Path,
    params: dict[str, Any],
) -> Path:
    """
    Fill a template file using params and write it to output_filename.
    """
    template_filename = Path(template_filename)
    output_filename = _ensure_parent_dir(output_filename)

    with open(template_filename, "r", encoding="utf-8") as f:
        src = Template(f.read())

    result = src.substitute(params)

    with open(output_filename, "w", encoding="utf-8") as f_out:
        f_out.write(result)

    return output_filename


def create_connection_from_template(
    name: str,
    state_dict: dict[str, Any],
    state_name: str,
    *,
    template_dir: str | Path,
    output_dir: str | Path,
    mask1: Any | None = None,
    mask2: Any | None = None,
) -> Path:
    """
    Extract a weight tensor from the state_dict, optionally apply output/input masks,
    and write a connection header file from template.
    """
    if state_name not in state_dict:
        raise KeyError(f"Missing state_dict key '{state_name}'.")

    weights = state_dict[state_name]
    if not isinstance(weights, torch.Tensor):
        weights = torch.as_tensor(weights)

    if weights.ndim != 2:
        raise ValueError(
            f"Connection tensor '{state_name}' must be 2D, got shape {tuple(weights.shape)}."
        )

    mask1_t = _to_bool_mask(mask1, weights.size(0), "mask1")
    mask2_t = _to_bool_mask(mask2, weights.size(1), "mask2")

    if mask1_t is not None:
        weights = weights[mask1_t, :]
    if mask2_t is not None:
        weights = weights[:, mask2_t]

    return create_connection_from_template_with_weights(
        name=name,
        weights=weights,
        template_dir=template_dir,
        output_dir=output_dir,
    )


def create_connection_from_template_with_weights(
    name: str,
    weights: torch.Tensor,
    *,
    template_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Write a connection header file from a given 2D weight tensor.
    """
    if not isinstance(weights, torch.Tensor):
        weights = torch.as_tensor(weights)

    if weights.ndim != 2:
        raise ValueError(
            f"Expected a 2D weights tensor for '{name}', got shape {tuple(weights.shape)}."
        )

    weights = weights.detach().cpu()

    output_size, input_size = weights.shape

    flat = [
        _format_c_float(weights[i, j].item())
        for i in range(output_size)
        for j in range(input_size)
    ]
    weights_string = "{" + ", ".join(flat) + "}"

    params = {
        "input_size": f"{input_size}",
        "output_size": f"{output_size}",
        "weights": weights_string,
        "name": name,
    }

    template = Path(template_dir) / "test_connection_file.templ"
    out = Path(output_dir) / f"test_controller_{name}_file.h"
    return create_from_template(template, out, params)


def create_neuron_from_template(
    name: str,
    state_dict: dict[str, Any],
    state_name: str,
    *,
    template_dir: str | Path,
    output_dir: str | Path,
    sigmoid: bool = False,
    mask: Any | None = None,
) -> Path:
    """
    Write a neuron header file from leak_i, leak_v, and thresh tensors.

    Expected keys:
    - {state_name}.leak_i
    - {state_name}.leak_v
    - {state_name}.thresh
    """
    leak_i_key = f"{state_name}.leak_i"
    leak_v_key = f"{state_name}.leak_v"
    thresh_key = f"{state_name}.thresh"

    for key in (leak_i_key, leak_v_key, thresh_key):
        if key not in state_dict:
            raise KeyError(f"Missing state_dict key '{key}'.")

    leak_i_tensor = state_dict[leak_i_key]
    leak_v_tensor = state_dict[leak_v_key]
    thresh_tensor = state_dict[thresh_key]

    if not isinstance(leak_i_tensor, torch.Tensor):
        leak_i_tensor = torch.as_tensor(leak_i_tensor)
    if not isinstance(leak_v_tensor, torch.Tensor):
        leak_v_tensor = torch.as_tensor(leak_v_tensor)
    if not isinstance(thresh_tensor, torch.Tensor):
        thresh_tensor = torch.as_tensor(thresh_tensor)

    leak_i_tensor = leak_i_tensor.detach().cpu().flatten()
    leak_v_tensor = leak_v_tensor.detach().cpu().flatten()
    thresh_tensor = thresh_tensor.detach().cpu().flatten()

    total_size = leak_i_tensor.numel()
    if leak_v_tensor.numel() != total_size or thresh_tensor.numel() != total_size:
        raise ValueError(
            f"Neuron tensor size mismatch for '{state_name}': "
            f"leak_i={leak_i_tensor.numel()}, "
            f"leak_v={leak_v_tensor.numel()}, "
            f"thresh={thresh_tensor.numel()}."
        )

    mask_t = _to_bool_mask(mask, total_size, "mask")

    d_i_vals: list[str] = []
    d_v_vals: list[str] = []
    th_vals: list[str] = []

    for i in range(total_size):
        if mask_t is not None and not bool(mask_t[i]):
            continue

        leak_i = leak_i_tensor[i]
        leak_v = leak_v_tensor[i]

        if sigmoid:
            leak_i = torch.sigmoid(leak_i)
            leak_v = torch.sigmoid(leak_v)
        else:
            leak_i = torch.clamp(leak_i, 0.0, 1.0)
            leak_v = torch.clamp(leak_v, 0.0, 1.0)

        th = torch.clamp(thresh_tensor[i], min=0.0)

        d_i_vals.append(_format_c_float(leak_i.item()))
        d_v_vals.append(_format_c_float(leak_v.item()))
        th_vals.append(_format_c_float(th.item()))

    hidden_size = len(d_i_vals)

    params = {
        "name": name,
        "hidden_size": f"{hidden_size}",
        "type": "1",
        "d_i": "{" + ", ".join(d_i_vals) + "}",
        "d_v": "{" + ", ".join(d_v_vals) + "}",
        "th": "{" + ", ".join(th_vals) + "}",
    }

    template = Path(template_dir) / "test_neuron_file.templ"
    out = Path(output_dir) / f"test_controller_{name}_file.h"
    return create_from_template(template, out, params)


def create_softreset_integrator_from_template(
    name: str,
    *,
    template_dir: str | Path,
    output_dir: str | Path,
    hidden_size: int = 4,
) -> Path:
    """
    Utility for generating a fixed soft-reset integrator neuron file.
    """
    d_i_vals = [_format_c_float(1.0)] * hidden_size
    d_v_vals = [_format_c_float(1.0)] * hidden_size
    th_vals = [_format_c_float(1.0)] * hidden_size

    params = {
        "name": name,
        "hidden_size": f"{hidden_size}",
        "type": "2",
        "d_i": "{" + ", ".join(d_i_vals) + "}",
        "d_v": "{" + ", ".join(d_v_vals) + "}",
        "th": "{" + ", ".join(th_vals) + "}",
    }

    template = Path(template_dir) / "test_neuron_file.templ"
    out = Path(output_dir) / "test_controller_integ_file.h"
    return create_from_template(template, out, params)