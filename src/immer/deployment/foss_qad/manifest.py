from ...runtimes.lfm.adapter import configured_model


def describe() -> dict[str, object]:
    return {"backend": "FOSS-QAD", "model": str(configured_model()) if configured_model() else None, "requires_acceptance": True}
