import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from chronos import Chronos2Pipeline


def load_model(model_name: str = "amazon/chronos-2"):
    print(f"[INFO] Loading Chronos-2 model: {model_name}")
    return Chronos2Pipeline.from_pretrained(model_name)
