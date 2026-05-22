<p align="center">
  <img src="Espikify_logo.jpg" alt="Espikify Logo" width="150">
</p>

<h1 align="center">Espikify</h1>

<p align="center">
  <b>Brew your PyTorch SNN models into ESP32-S3-ready C headers.</b>
</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  </a>
  <img src="https://img.shields.io/badge/target-ESP32--S3-green.svg" alt="Target: ESP32-S3">
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#cli-usage">CLI Usage</a> •
  <a href="#generated-output">Generated Output</a> •
  <a href="#examples">Examples</a> •
  <a href="#esp32-test-examples">ESP32 Tests</a> •
  <a href="#citation">Citation</a>
</p>

## ✨ Features

- **Seamless Transformation**: Export trained PyTorch SNN attitude and control checkpoints into ESP32-S3-ready C/header files.
- **ESP32-S3 Ready**: Generate controller headers and runtime files for ESP32-S3 firmware projects.
- **Different Runtime Backends**: Choose between `optimized` and `event-driven` runtime modes.
- **Flexible Model Inputs**: Use direct `.pt` files, folders containing `model.pt`, or Weights & Biases run names.
- **Controller Header Generation**: Generate the main controller configuration header and controller component headers.
- **Batch Processing**: Convert the attitude and control networks with a single command.

## 🚀 Quick Start

### Installation

```bash
git clone <repo-url>
cd espikify
python -m pip install -e .
```

Package requirements:

| Requirement | Version |
|---|---|
| Python | `>=3.10` |
| `numpy` | `>=1.24` |
| `torch` | `>=2.0` |
| `wandb` | `>=0.16` |

Check the CLI:

```bash
espikify --help
```

## CLI usage

```bash
espikify \
  --att-model ATT_MODEL \
  --control-model CONTROL_MODEL \
  --mode {optimized,event-driven} \
  [--output-dir DIR] \
  [--wandb-project PROJECT] \
  [--models-dir DIR] \
  [--att-mask CSV] \
  [--control-mask CSV] \
  [--no-copy-runtime]
```

## Generated output

An export writes:

```text
<output-dir>/
├── test_controller_fwr_conf.h
├── controller/
│   └── test_controller_*_file.h
└── runtime/              # omitted with --no-copy-runtime
```

## Examples

### Export in optimized mode

```bash
espikify \
  --att-model ./models/attitude/model.pt \
  --control-model ./models/control \
  --mode optimized \
  --output-dir ./build/optimized_export
```

### Export in event-driven mode

```bash
espikify \
  --att-model ./models/attitude \
  --control-model ./models/control \
  --mode event-driven \
  --output-dir ./build/event_export
```

## ESP32 test examples

The repository includes two Arduino sketches for testing generated controller exports:

| Sketch | Use with |
|---|---|
| `optimized-snn.ino` | `--mode optimized` |
| `snn-event-driven.ino` | `--mode event-driven` |

Use the sketch that matches the selected export mode. Both sketches load `test_controller_fwr_conf.h`, build a `NetworkController`, run `forward_network`, and stream controller output over serial. The event-driven sketch also uses the event-driven runtime path and reports spike-count statistics.

## Acknowledgements

Espikify stands on the shoulders of giants. We are grateful to these projects and communities:

- **[TinySNN](https://github.com/Huizerd/tinysnn)** — MIT Licensed. This work inspired the project.
- **Espressif Systems** — for the ESP32-S3 platform and ecosystem.

Contributions and issue reports are welcome.


