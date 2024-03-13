from typing import Any, Dict
import yaml
from pathlib import Path
from self_query_summarization.config.config import ROOT_DIR


def build_path(dir_path: str, file_path: str) -> Path:
    return ROOT_DIR / dir_path / file_path


def load_config_yaml(config_path: str) -> Dict[Any,Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)