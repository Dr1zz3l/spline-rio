"""Load all YAML config files from config/ and return as a dict-of-dicts."""
import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "config"


def load_config():
    """Load all YAML configs and return as a dict-of-dicts keyed by filename stem."""
    result = {}
    for f in sorted(_CONFIG_DIR.glob("*.yaml")):
        with open(f) as fh:
            result[f.stem] = yaml.safe_load(fh)
    return result
