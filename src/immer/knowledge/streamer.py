"""Streamer: Weltwissen aus HF-Safetensors, seitenweise gestreamt.

Der gemessene Moonshot (FORMEL-FUNDAMENT §4): Header-only-Inventar,
Range-GETs vom CDN, CAS-Hashes. 27B-Scan = 0,21 % Transfer.
Diese Fassade macht daraus ZWEI Aufrufe:

    s = Streamer("Qwen/Qwen2.5-0.5B")
    inv   = s.inventory()            # komplettes Tensor-Verzeichnis
    rows  = s.rows("model.safetensors", "model.embed_tokens.weight", 0, 8)

Byte-Budget wird mitgezählt — die Zahl, die die Demo trägt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "mitglm"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import hf_organ_reader as hor  # noqa: E402
import casi_tensor_map as ctm  # noqa: E402


class Streamer:
    def __init__(self, repo_id: str, revision: str = "main", budget_mb: float = 200.0) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.budget = hor.Budget(budget_mb)
        self._reader: Any = None
        self._inventory: dict[str, Any] | None = None

    @property
    def reader(self) -> Any:
        if self._reader is None:
            self._reader = hor.HFRangeReader(self.repo_id, revision=self.revision)
            self._reader.budget = self.budget
        return self._reader

    def inventory(self) -> dict[str, Any]:
        """Header-only-Inventar ohne Payload (der billige Teil)."""
        if self._inventory is None:
            self._inventory = hor.scan_inventory(self.reader, budget=self.budget)
        return self._inventory

    def tensors(self) -> list[dict[str, Any]]:
        """Flache Liste aller Tensor-Einträge aus dem Inventar."""
        return list(self.inventory().get("tensors", []))

    def find(self, name: str) -> dict[str, Any]:
        for entry in self.tensors():
            if entry["name"] == name:
                return entry
        raise KeyError(f"Tensor {name!r} nicht im Inventar von {self.repo_id}")

    _ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}

    def rows(self, tensor_name: str, start_row: int = 0, n_rows: int = 8,
             n_blocks: int = 8) -> Any:
        """Gestreamte Zeilen eines Tensors als numpy-Array (bf16→f32)."""
        import numpy as np

        meta = self.find(tensor_name)
        shape = meta["shape"]
        if len(shape) < 2:
            raise ValueError("rows() braucht 2D-Tensoren")
        n_cols = int(shape[1])
        itemsize = self._ITEMSIZE.get(str(meta["dtype"]).upper())
        if itemsize is None:
            raise ValueError(f"unbekanntes dtype {meta['dtype']}")
        off_b, _off_e = meta["offset_in_shard"]
        row_bytes = n_cols * itemsize
        raw, _reqs, got = hor.fetch_blocks(
            self.reader, meta["shard"], meta["data_start"],
            off_b + start_row * row_bytes, n_rows, n_cols, itemsize,
            cap=row_bytes * max(n_rows, 1), name=tensor_name, n_blocks=n_blocks,
            rows_exact=n_rows,
        )
        if str(meta["dtype"]).upper() == "BF16":
            return ctm.bf16_rows_to_f32(raw.view(np.uint16), (got, n_cols))
        return np.frombuffer(raw[: got * row_bytes], dtype=np.float32).reshape(got, n_cols)

    def budget_line(self) -> str:
        return f"{self.budget.body / 1048576:.3f} MB body / {self.budget.requests} reqs"

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        return self.reader.get_range(shard, offset, offset + length)

    def bytes_moved(self) -> int:
        counter = getattr(self.reader, "bytes_downloaded", None)
        if callable(counter):
            return int(counter())
        stats = getattr(self.reader, "stats", None)
        if isinstance(stats, dict):
            return int(stats.get("bytes", 0))
        total = getattr(self.reader, "total_bytes", None)
        return int(total) if total else -1


def available() -> bool:
    try:
        import requests  # noqa: F401
        _ = hor.HFRangeReader
        return True
    except Exception:
        return False
