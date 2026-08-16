# LED-ML

**LED-ML** (Layer Extraction & Diagnostics with Machine Learning) is a lightweight runtime diagnostics package for large language models. It attaches to model layers, captures activation and gradient signals, and exposes structured telemetry for analysis, latency profiling, RAM footprint checks, and visualization.

## Features

- **Runtime Diagnostics**: Attach hooks to LLM layers and capture activation/gradient signals
- **Latency Profiling**: Measure and analyze hook execution latency
- **Memory Analysis**: Calculate RAM footprint of captured data
- **Visualization**: Generate ASCII text visualizations and GIF-based layer maps
- **Model Support**: Works with popular models via the `transformers` library

## Package Contents

The `led_ml` package includes:

- `led_core.py` — Main diagnostics engine for model layer instrumentation
- `led_latency.py` — Latency measurement and analysis utilities
- `led_size.py` — Memory footprint calculation
- `led_viz.py` — Text and GIF visualization tools
- `supported_models.json` — Configuration for supported model families

## Runtime Dependencies

The package requires:

- `torch>=1.9.0`
- `transformers>=4.0.0`
- `matplotlib>=3.5.0`
- `pillow>=9.0.0`

These are automatically installed with `pip install led-ml`.

## Quick Start

Here's a basic example of using LED-ML to diagnose a language model:

```python
import json
import led_ml
from led_ml import led_viz, led_size, led_latency

# Initialize LED diagnostics
led = led_ml.led_core(mode="trajectory")

# Get list of supported models
print("Supported models:", led.get_supported_models())

# Run inference on a model and capture diagnostics
prompt = "What is the color of chlorophyll?"
response = led.run_model(prompt, model_family="qwen2")
print(f"Model response: {response}")

# Extract captured diagnostics
diagnostics = led.get_led()

# Analyze memory footprint
ram_bytes = led_size.ram_led(diagnostics)
ram_mb = ram_bytes / (1024.0 * 1024.0)
print(f"Diagnostic data size: {ram_mb:.2f} MB")

# Analyze latency
latency_stats = led_latency.lat_led(diagnostics)

# Generate visualizations
led_viz.text_led(diagnostics)    # Text-based layer activation map
led_viz.gif_led(diagnostics)     # Generate animated GIF

# Save diagnostics to file
model_name = diagnostics["model"].replace("/", "-")
with open(f"{model_name}_diagnostics.json", "w") as f:
    json.dump(diagnostics, f)
```

## Text Visualization Example

LED-ML generates text-based visualizations showing activation patterns across model layers:

```
Model: Qwen/Qwen2.5-7B | Layers: 28 | Mode: trajectory
Prompt: What is the color of chlorophyll?
Response: Chlorophyll is green in color.

Layer | Activation Map (First 10%)
------+--------------------------------------------------
L00   | ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
L01   | ░░░░░░░░░░▲░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
L02   | ░░░░░░░░░░▲░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
...
```

## Use Cases

- **Performance Profiling**: Identify latency bottlenecks in model inference
- **Memory Analysis**: Understand memory requirements of activation data
- **Model Inspection**: Visualize information flow through model layers
- **Research**: Analyze model behavior and layer importance

## License

See LICENSE file for details.
