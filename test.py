import led_ml
import pdb
import json 
from led_ml import led_viz
from led_ml import led_size
from led_ml import led_latency

prompt = "what is color of chlorophyll?"

# initialise LED object
obj = led_ml.led_core(mode="trajectory")

print(f"Models supported by package, {obj.get_supported_models()}")

# Run the supported model
answer = obj.run_model(prompt, model_family="qwen2")
print(f"Response is : {answer}")

# Get dump and update the json
led_data = obj.get_led()

# Save to json
model_name = led_data["model"].strip()
model_name = model_name.replace("/", "-")

json_f = f"{model_name}.json"
with open(json_f, 'w') as f:
    json.dump(led_data, f)

# LED metrics in RAM
ram_bytes = led_size.ram_led(led_data)
ram_mb = ram_bytes / (1024.0 * 1024.0)

print(f"True RAM Memory Footprint of raw object: {ram_bytes} bytes ({ram_mb:.2f} MB)")

# Calculate hook latency from RAM object
_ = led_latency.lat_led(led_data)

# Load from json
with open(json_f) as f:
    f_data = json.load(f)

# LED metrics in RAM
ram_bytes = led_size.ram_led(f_data)
ram_mb = ram_bytes / (1024.0 * 1024.0)

print(f"True RAM Memory Footprint of object from json file : {ram_bytes} bytes ({ram_mb:.2f} MB)")

# Calculate hook latency
_ = led_latency.lat_led(f_data)

# Generate visual gif
led_viz.gif_led(f_data)

# text visualisation
led_viz.text_led(f_data)
