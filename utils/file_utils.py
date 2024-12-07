# utils/file_utils.py
from pathlib import Path

from huggingface_hub import hf_hub_download

from config.settings import HF_REPO_ID, LOCAL_DIR, PHOTOMAKER_FILENAME


def ensure_photomaker_model() -> Path:
    """Ensure PhotoMaker model exists locally or download it."""
    model_path = LOCAL_DIR / PHOTOMAKER_FILENAME
    if not model_path.exists():
        return Path(
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=PHOTOMAKER_FILENAME, repo_type="model", local_dir=LOCAL_DIR
            )
        )
    return model_path
