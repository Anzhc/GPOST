import re
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download


def _extract_version(name: str) -> Optional[int]:
    m = re.search(r"text-seg-v(\d+)", name)
    return int(m.group(1)) if m else None


def _latest_local_model(models_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
    best_path = None
    best_ver = -1
    for p in models_dir.glob("*"):
        ver = _extract_version(p.name)
        if ver is not None and ver > best_ver:
            best_ver = ver
            best_path = p
    if best_ver == -1:
        return None, None
    return best_path, best_ver


def _latest_remote_model(repo_id: str) -> Tuple[Optional[str], Optional[int]]:
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        print(f"[model_helper] Failed to list repo files: {e}")
        return None, None
    best_file = None
    best_ver = -1
    for f in files:
        ver = _extract_version(f)
        if ver is not None and ver > best_ver:
            best_ver = ver
            best_file = f
    if best_ver == -1:
        return None, None
    return best_file, best_ver


def prepare_latest_model(repo_id: str = "Anzhc/Anzhcs_YOLOs", models_subdir: str = "models") -> Optional[Path]:
    """Ensure latest YOLO model is downloaded and return its local path."""
    base = Path(__file__).resolve().parent
    models_dir = base / models_subdir
    models_dir.mkdir(exist_ok=True)

    local_path, local_ver = _latest_local_model(models_dir)
    if local_ver is not None:
        print(f"[model_helper] Local model version: v{local_ver} ({local_path.name})")
    else:
        print("[model_helper] No local model found.")

    remote_file, remote_ver = _latest_remote_model(repo_id)
    if remote_ver is None:
        print("[model_helper] Could not determine latest model from HuggingFace.")
        return local_path
    print(f"[model_helper] Latest repo version: v{remote_ver} ({remote_file})")

    if local_ver is None or remote_ver > local_ver:
        print("[model_helper] Downloading newer model...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=remote_file,
                local_dir=models_dir,
                resume_download=True,
            )
            local_path = Path(downloaded)
            print(f"[model_helper] Downloaded {local_path.name}")
        except Exception as e:
            print(f"[model_helper] Download failed: {e}")
            return local_path
    else:
        print("[model_helper] Local model is up to date.")
    return local_path
