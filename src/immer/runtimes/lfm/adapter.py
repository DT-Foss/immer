from pathlib import Path


def configured_model() -> Path | None:
    """Return the optional model path without loading it."""
    import os

    value = os.environ.get("IMMER_LFM_MODEL")
    return Path(value).expanduser() if value else None
