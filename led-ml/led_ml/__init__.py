"""LED-ML Package: Layer Extraction & Diagnostics with Machine Learning"""

import json
from pathlib import Path
from importlib import resources
from .led_core import led_core


def load_supported_models():
    """Load supported models configuration from package resource."""
    try:
        # Python 3.9+: Use importlib.resources.files
        if hasattr(resources, 'files'):
            config_file = resources.files(__package__).joinpath('supported_models.json')
            config_data = config_file.read_text(encoding='utf-8')
        else:
            # Fallback for Python 3.7-3.8
            with resources.open_text(__package__, 'supported_models.json') as f:
                config_data = f.read()
        
        return json.loads(config_data)
    except FileNotFoundError:
        # Fallback: read from file system if resource not found
        config_path = Path(__file__).parent / 'supported_models.json'
        with open(config_path, 'r') as f:
            return json.load(f)


# Make supported models available at package level
SUPPORTED_MODELS = load_supported_models()

__all__ = ['SUPPORTED_MODELS', 'load_supported_models']
