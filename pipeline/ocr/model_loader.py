import importlib.util
import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"


def _resolve_model_path(model_name: str) -> Path:
    resolved = (MODELS_DIR / model_name).resolve()
    if not str(resolved).startswith(str(MODELS_DIR.resolve())):
        raise ValueError(f"Invalid model name: '{model_name}' (path traversal detected)")
    return resolved


def load_model_config(model_name: str) -> dict:
    model_dir = _resolve_model_path(model_name)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found for model '{model_name}' at {config_path}"
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_model_prompts(model_name: str) -> dict:
    model_dir = _resolve_model_path(model_name)
    prompts_path = model_dir / "prompts.py"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"prompts.py not found for model '{model_name}' at {prompts_path}"
        )
    spec = importlib.util.spec_from_file_location(
        f"models.{model_name}.prompts", str(prompts_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PROMPTS"):
        raise ValueError(
            f"Model '{model_name}' prompts.py must define PROMPTS dict"
        )
    return module.PROMPTS
