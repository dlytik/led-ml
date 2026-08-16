# LED-ML Package Setup

## Overview
This repo contains an installable Python package in the `led_ml/` directory and a standalone example script at the repo root: `test.py`. The script is for demonstration and validation, not a package module, so it is intentionally excluded from package installation metadata.

## Current package layout
```text
led-ml/
├── led_ml/
│   ├── __init__.py
│   ├── constants.py
│   ├── led_core.py
│   ├── led_latency.py
│   ├── led_size.py
│   ├── led_viz.py
│   └── supported_models.json
├── setup.py
├── MANIFEST.in
├── PACKAGE_SETUP.md
├── README.md
├── test.py
└── ...
```

## Runtime dependencies
The package imports these dependencies at runtime:
- `torch`
- `transformers`
- `matplotlib`
- `pillow`

Standard-library modules such as `json`, `sys`, `pathlib`, and `statistics` are not package dependencies.

## Installation
From the package directory:

```bash
cd led-ml
pip install -e .
```

Or for a normal install:

```bash
cd led-ml
pip install .
```

## Example usage
The repo-root script `test.py` shows the intended package workflow:

```python
import led_ml
import led_size
import led_latency
import led_viz

prompt = "what is color of chlorophyll?"
obj = led_ml.led_core(mode="trajectory")
print(obj.get_supported_models())

answer = obj.run_model(prompt, model_family="openchat")
print(f"Response is : {answer}")

led_data = obj.get_led()

ram_bytes = led_size.ram_led(led_data)
_ = led_latency.lat_led(led_data)
led_viz.gif_led(led_data)
led_viz.text_led(f_data)
```

## Resource loading
The package reads `supported_models.json` using `importlib.resources` and falls back to direct file access if needed. This is handled by `led_ml/__init__.py` and the loader used by `led_core`.

## Notes
- `test.py` is a repo-local example and not installed as part of the package.
- Package setup metadata should reflect only the installable package under `led_ml/`, not the standalone example script.
