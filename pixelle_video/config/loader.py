# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration loader - Pure YAML

Handles loading and saving configuration from/to YAML files.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from loguru import logger

# ${VAR} or ${VAR:-default}; only whole-string references are expanded
ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")

EnvRefs = Dict[Tuple[str, ...], str]


def expand_env_refs(data: Any, path: Tuple[str, ...] = (), refs: Optional[EnvRefs] = None) -> Tuple[Any, EnvRefs]:
    """
    Replace string values of the form ${VAR} / ${VAR:-default} with environment values.

    Lets config.yaml reference secrets (e.g. instabox Org Secret placeholders or a
    chmod-600 EnvironmentFile) instead of storing them. Returns the expanded data and
    a map of key paths to their original references, so saving can write the
    reference back instead of the secret.
    """
    refs = {} if refs is None else refs
    if isinstance(data, dict):
        return {k: expand_env_refs(v, path + (str(k),), refs)[0] for k, v in data.items()}, refs
    if isinstance(data, str):
        match = ENV_REF_PATTERN.match(data.strip())
        if match:
            name, default = match.group(1), match.group(2)
            value = os.environ.get(name)
            if value is None:
                if default is None:
                    logger.warning(f"Config references unset environment variable {name} at {'.'.join(path)}")
                value = default or ""
            refs[path] = data.strip()
            return value, refs
    return data, refs


def restore_env_refs(data: Any, refs: EnvRefs, path: Tuple[str, ...] = ()) -> Any:
    """Inverse of expand_env_refs for saving: put ${VAR} references back."""
    if isinstance(data, dict):
        return {k: restore_env_refs(v, refs, path + (str(k),)) for k, v in data.items()}
    if path in refs:
        ref = refs[path]
        match = ENV_REF_PATTERN.match(ref)
        current = os.environ.get(match.group(1), match.group(2) or "") if match else None
        # Keep the reference only if the value is unchanged; an edit (including
        # clearing the field) is an explicit user choice and is saved as-is
        if data == current:
            return ref
    return data


def load_config_dict(config_path: str = "config.yaml", refs: Optional[EnvRefs] = None) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        data, _ = expand_env_refs(data, refs=refs)
        logger.info(f"Configuration loaded from {config_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def save_config_dict(config: dict, config_path: str = "config.yaml", refs: Optional[EnvRefs] = None):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to config file
    """
    if refs:
        config = restore_env_refs(config, refs)
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise

