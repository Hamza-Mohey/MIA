"""Download the optimized Whisper Medium representation once for offline use."""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from modules.config import FASTER_WHISPER_MODEL_DIR


def main() -> None:
    path = snapshot_download(
        "Systran/faster-whisper-medium",
        local_dir=str(FASTER_WHISPER_MODEL_DIR),
    )
    print(f"Faster-Whisper Medium cached at {path}")


if __name__ == "__main__":
    main()
