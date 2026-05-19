from __future__ import annotations

import argparse
import sys
from pathlib import Path

from espikify.converter import convert_models


REQUIRED_TEMPLATES = [
    "test_connection_file.templ",
    "test_controller_fwr_conf.templ",
    "test_li_out_file.templ",
    "test_neuron_file.templ",
]


def _check_package_resources(package_root: Path) -> tuple[Path, Path]:
    template_dir = package_root / "templates"
    runtime_root = package_root / "runtimes"

    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    missing_templates = [name for name in REQUIRED_TEMPLATES if not (template_dir / name).exists()]
    if missing_templates:
        raise FileNotFoundError(
            "Missing template file(s): " + ", ".join(str(template_dir / name) for name in missing_templates)
        )

    if not runtime_root.exists():
        raise FileNotFoundError(f"Runtime root directory not found: {runtime_root}")

    for subdir in ["optimized", "event_driven"]:
        p = runtime_root / subdir
        if not p.exists():
            raise FileNotFoundError(f"Missing runtime backend directory: {p}")

    return template_dir, runtime_root


def _build_parser() -> argparse.ArgumentParser:
    examples = """
Accepted model inputs:
  1) Direct .pt file path
     espikify --att-model ./models/att/model.pt --control-model ./models/ctrl/model.pt --mode optimized

  2) Folder containing model.pt
     espikify --att-model ./models/att_run --control-model ./models/ctrl_run --mode optimized

  3) W&B run name (requires --wandb-project)
     espikify --att-model olive-firebrand-195 --control-model rich-snowball-319 --mode optimized --wandb-project neuro-1

More examples:
  espikify --att-model ./attitude --control-model ./control --mode event-driven --output-dir ./build/export
  espikify --att-model runA --control-model runB --mode optimized --wandb-project neuro-1 --models-dir ./param/models
"""

    parser = argparse.ArgumentParser(
        prog="espikify",
        description=(
            "Convert PyTorch SNN attitude/control models into ESP32-ready C headers "
            "and copy the selected runtime backend."
        ),
        epilog=examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--att-model",
        required=True,
        metavar="ATT_MODEL",
        help=(
            "Attitude model input. Accepted forms:\n"
            "  - direct path to model.pt\n"
            "  - path to a folder containing model.pt\n"
            "  - W&B run name, if --wandb-project is provided"
        ),
    )

    parser.add_argument(
        "--control-model",
        required=True,
        metavar="CONTROL_MODEL",
        help=(
            "Control model input. Accepted forms:\n"
            "  - direct path to model.pt\n"
            "  - path to a folder containing model.pt\n"
            "  - W&B run name, if --wandb-project is provided"
        ),
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["optimized", "event-driven"],
        help="Runtime backend to export.",
    )

    parser.add_argument(
        "--output-dir",
        default="./build/espikify_export",
        metavar="DIR",
        help="Directory where generated files will be written.",
    )

    parser.add_argument(
        "--wandb-project",
        default=None,
        metavar="PROJECT",
        help=(
            "Weights & Biases project path used when a model input is a W&B run name, "
            'for example "neuro-1" or "entity/project".'
        ),
    )

    parser.add_argument(
        "--models-dir",
        default="param/models",
        metavar="DIR",
        help=(
            "Local cache directory for W&B-downloaded models. "
            "Used only when --wandb-project is provided."
        ),
    )

    parser.add_argument(
        "--att-mask",
        default=None,
        metavar="CSV",
        help="Optional CSV mask file for the attitude network.",
    )

    parser.add_argument(
        "--control-mask",
        default=None,
        metavar="CSV",
        help="Optional CSV mask file for the control network.",
    )

    parser.add_argument(
        "--no-copy-runtime",
        action="store_true",
        help="Generate headers only and do not copy runtime C/H files.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        package_root = Path(__file__).resolve().parent
        template_dir, runtime_root = _check_package_resources(package_root)

        output_dir = Path(args.output_dir).expanduser().resolve()

        result = convert_models(
            att_model=args.att_model,
            control_model=args.control_model,
            mode=args.mode,
            output_dir=output_dir,
            template_dir=template_dir,
            runtime_root=runtime_root,
            att_mask_path=args.att_mask,
            control_mask_path=args.control_mask,
            wandb_project=args.wandb_project,
            models_dir=args.models_dir,
            copy_runtime=not args.no_copy_runtime,
        )

        print("\nExport completed successfully.")
        print(f"Config header: {result['config_header']}")
        print(f"Controller headers: {result['controller_dir']}")
        if not args.no_copy_runtime:
            print(f"Runtime files: {result['output_dir'] / 'runtime'}")

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()