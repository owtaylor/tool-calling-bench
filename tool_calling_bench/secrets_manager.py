# secrets_manager.py
import yaml
import os
import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

SECRETS_DIR = Path.home() / ".config" / "tool-calling-bench"
SECRETS_FILE_PATH = SECRETS_DIR / "secrets.yaml"


class Secrets:
    """
    Manages loading secrets from a YAML file (~/.config/tool-calling-bench/secrets.yaml)
    with a fallback to environment variables.
    """

    def __init__(self):
        self._secrets_data = {}
        self._load_secrets()

    def _load_secrets(self):
        """Loads secrets from the YAML file if it exists."""
        if not SECRETS_DIR.exists():
            try:
                # Attempt to create the directory if it doesn't exist
                SECRETS_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created secrets directory: {SECRETS_DIR}")
            except OSError as e:
                logger.warning(
                    f"Could not create secrets directory {SECRETS_DIR}: {e}. Will rely solely on environment variables."
                )
                return  # Cannot proceed with file loading if dir creation failed

        if SECRETS_FILE_PATH.exists() and SECRETS_FILE_PATH.is_file():
            try:
                with open(SECRETS_FILE_PATH, "r", encoding="utf-8") as f:
                    loaded_data = yaml.safe_load(f)
                    if isinstance(loaded_data, dict):
                        self._secrets_data = loaded_data
                        logger.info(f"Loaded secrets from {SECRETS_FILE_PATH}")
                    else:
                        logger.warning(
                            f"Secrets file {SECRETS_FILE_PATH} does not contain a valid YAML dictionary. Ignoring."
                        )
            except yaml.YAMLError as e:
                logger.error(f"Error parsing secrets file {SECRETS_FILE_PATH}: {e}. Ignoring file.")
            except Exception as e:
                logger.error(f"Error reading secrets file {SECRETS_FILE_PATH}: {e}. Ignoring file.")
        else:
            logger.info(
                f"Secrets file not found at {SECRETS_FILE_PATH}. Will rely on environment variables."
            )
            # Optionally, create an empty file here if desired
            # try:
            #     with open(SECRETS_FILE_PATH, 'w') as f:
            #         f.write("# Tool Calling Bench Secrets\n")
            #         f.write("# Add secrets here in KEY: VALUE format\n")
            #     logger.info(f"Created empty secrets file template at {SECRETS_FILE_PATH}")
            # except OSError as e:
            #      logger.warning(f"Could not create empty secrets file at {SECRETS_FILE_PATH}: {e}")

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Gets a secret value.

        Checks the loaded secrets file first, then falls back to environment variables.

        Args:
            key: The name of the secret (e.g., "ANTHROPIC_API_KEY").
            default: The value to return if the key is not found anywhere.

        Returns:
            The secret value or the default.
        """
        # 1. Check secrets file data
        value = self._secrets_data.get(key)
        if value is not None:
            logger.debug(f"Retrieved secret '{key}' from secrets file.")
            return value

        # 2. Check environment variables
        value = os.getenv(key)
        if value is not None:
            logger.debug(f"Retrieved secret '{key}' from environment variables.")
            return value

        # 3. Return default
        logger.debug(f"Secret '{key}' not found in file or environment. Returning default.")
        return default
