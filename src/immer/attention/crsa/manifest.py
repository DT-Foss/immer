from ...runtimes.lfm.adapter import configured_model


def describe() -> dict[str, object]:
    return {"backend": "CRSA", "model": str(configured_model()) if configured_model() else None, "weight_transfer": "none"}
