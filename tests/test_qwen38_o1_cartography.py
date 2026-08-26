from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

from immer.runtimes.ooe.crystal import CrystalTamperError
from immer.runtimes.qwen3_8 import (
    AtlasQueryResult,
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    RuntimeProvenance,
    WeightCoordinate,
    prompt_token_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_o1_cartography.py"
CODE_REVISION = "a" * 40


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen38_o1_cartography", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen3.8 O1 cartography CLI")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cartography = _load_script()


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _FakeStream:
    def __init__(self, **_kwargs) -> None:
        self.loss_ema = 0.0
        self.tokens = 0

    def observe(self, value: str) -> None:
        self.tokens += len(value.encode("utf-8"))

    def snapshot(self):
        return {"loss_ema": self.loss_ema, "tokens": self.tokens}

    def restore(self, state) -> None:
        self.loss_ema = float(state["loss_ema"])
        self.tokens = int(state["tokens"])


@dataclass(frozen=True)
class _FakeCoordinate:
    layer: int
    module: str
    tensor: str
    head_index: int | None = None
    row_start: int | None = 0
    row_end: int | None = 2
    tensor_absolute_offset: int = 0
    tensor_length: int = 2
    range_absolute_offset: int = 0
    range_length: int = 2

    @property
    def sha256(self) -> str:
        return _digest(
            {"layer": self.layer, "module": self.module, "tensor": self.tensor}
        )


@dataclass(frozen=True)
class _FakeIntervention:
    mode: str
    configuration_sha256: str


@dataclass(frozen=True)
class _FakeMeasurement:
    model_pin: ModelPin
    probe: object
    coordinate: _FakeCoordinate
    intervention: _FakeIntervention
    spec_sha256: str
    observed_semantic_label: str | None = None

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "coordinate_sha256": self.coordinate.sha256,
                "intervention": self.intervention.mode,
                "model_pin_sha256": self.model_pin.sha256,
                "probe_sha256": self.probe.sha256,
                "spec_sha256": self.spec_sha256,
                "semantic_label": self.observed_semantic_label,
            }
        )

    def to_document(self):
        return {
            "coordinate_sha256": self.coordinate.sha256,
            "intervention": self.intervention.mode,
            "model_pin_sha256": self.model_pin.sha256,
            "probe_sha256": self.probe.sha256,
            "sha256": self.sha256,
            "semantic_label": self.observed_semantic_label,
        }


@dataclass(frozen=True)
class _FakeAppendReceipt:
    record_kind: str
    record_sha256: str
    segment_sha256: str
    appended: bool


class _FakeResult:
    def __init__(self, measurement: object) -> None:
        self.measurement = measurement
        self.measurements_in_append_order = (measurement,)
        operation = SimpleNamespace(source_bytes=7)
        self.access_trace = SimpleNamespace(operations=(operation,))
        self.control_access_trace = None
        # Two logical leaves carry the same operation-level accounting.  The
        # scheduler must charge the operation once, not once per leaf.
        leaf = SimpleNamespace(source_bytes=7)
        self.tensor_range_receipts = (leaf, leaf)
        self.control_tensor_range_receipts = ()

    def verify(self) -> None:
        if not self.measurement.sha256:
            raise AssertionError("unsealed fake measurement")


class _FakeAtlas:
    stores: dict[str, dict[str, object]] = {}
    histories: dict[str, list[GraphRevision]] = {}

    def __init__(self, path, *, model_pin, tensor_plans) -> None:
        self.path = str(Path(path).absolute())
        self.model_pin = model_pin
        if not tensor_plans:
            raise AssertionError("atlas requires injected plans")
        self.store = self.stores.setdefault(self.path, {})
        self.history = self.histories.setdefault(
            self.path,
            [GraphRevision(0, _digest([]))],
        )

    def verify_or_raise(self) -> bool:
        if any(value.model_pin != self.model_pin for value in self.store.values()):
            raise AssertionError("fake atlas pin tamper")
        return True

    def revision(self) -> GraphRevision:
        return self.history[-1]

    def revision_history(self) -> tuple[GraphRevision, ...]:
        return tuple(self.history)

    def contains_revision(self, revision: GraphRevision) -> bool:
        return any(value == revision for value in self.history)

    def append_measurement(self, measurement):
        prior = self.store.get(measurement.sha256)
        if prior is not None and prior != measurement:
            raise AssertionError("fake atlas identity collision")
        appended = prior is None
        self.store[measurement.sha256] = measurement
        if appended:
            self.history.append(
                GraphRevision(
                    sequence=self.history[-1].sequence + 1,
                    event_sha256=_digest(sorted(self.store)),
                )
            )
        return _FakeAppendReceipt(
            record_kind="measurement",
            record_sha256=measurement.sha256,
            segment_sha256=_digest({"measurement": measurement.sha256}),
            appended=appended,
        )

    def _result(self, values):
        return AtlasQueryResult(
            measurements=tuple(sorted(values, key=lambda row: row.sha256)),
            promotions=(),
            replicas=(),
        )

    def query_by_prompt_signature(self, prompt_sha256):
        return self._result(
            value
            for value in self.store.values()
            if value.probe.prompt_signature == prompt_sha256
        )

    def query_by_coordinate(self, coordinate_sha256):
        return self._result(
            value
            for value in self.store.values()
            if value.coordinate.sha256 == coordinate_sha256
        )

    def query_by_semantic_label(self, semantic_label):
        return self._result(
            value
            for value in self.store.values()
            if value.observed_semantic_label == semantic_label
        )

    def coverage_matrix(self):
        return {
            "measurement_count": len(self.store),
            "measurement_sha256s": sorted(self.store),
        }


class _FakeModel:
    def __init__(self, model_pin: ModelPin) -> None:
        self.model_pin = model_pin


class _FakeProbe:
    calls: list[str] = []
    failures_remaining = 0

    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def execute(self, spec, *, atlas_head_revision):
        if not isinstance(atlas_head_revision, GraphRevision):
            raise AssertionError("probe did not receive a real GraphRevision")
        self.calls.append(spec.sha256)
        if self.failures_remaining:
            type(self).failures_remaining -= 1
            raise RuntimeError("injected probe failure")
        coordinate = _FakeCoordinate(
            layer=spec.coordinate.layer,
            module=spec.coordinate.module,
            tensor=spec.coordinate.tensor,
        )
        return _FakeResult(
            _FakeMeasurement(
                model_pin=self.model.model_pin,
                probe=spec.probe_identity,
                coordinate=coordinate,
                intervention=_FakeIntervention(
                    str(spec.intervention_mode),
                    _digest({"alpha": 0.0, "kind": "original-qwen-identity"}),
                ),
                spec_sha256=spec.sha256,
                observed_semantic_label=spec.semantic_label,
            )
        )


_WEIGHT_REVISION = GraphRevision(7, _digest({"graph": "qwen-weight-rail"}))
_RUNTIME_PROVENANCE = RuntimeProvenance(
    code_revision=CODE_REVISION,
    source_manifest_sha256=_digest({"source": "local-causal-bundle"}),
    dependency_manifest_sha256=_digest({"dependencies": "fixture"}),
    runtime_configuration_sha256=_digest({"runtime": "fixture"}),
    platform_sha256=_digest({"platform": "fixture"}),
)


class _AuthenticProbe:
    calls: list[str] = []

    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def execute(self, spec, *, atlas_head_revision):
        if not isinstance(atlas_head_revision, GraphRevision):
            raise AssertionError("probe did not receive the Atlas graph head")
        self.calls.append(spec.sha256)
        coordinate = WeightCoordinate(
            layer=spec.coordinate.layer,
            module=spec.coordinate.module,
            tensor=spec.coordinate.tensor,
            dtype="BF16",
            shape=(2, 1),
            shard=f"layer-{spec.coordinate.layer}.safetensors",
            tensor_absolute_offset=spec.coordinate.layer * 16,
            tensor_length=4,
            range_absolute_offset=spec.coordinate.layer * 16,
            range_length=4,
            row_start=0,
            row_end=2,
        )
        signal = float(spec.coordinate.layer + 1)
        measurement = MeasurementReceipt(
            model_pin=self.model.model_pin,
            coordinate=coordinate,
            probe=spec.probe_identity,
            intervention=InterventionIdentity(
                mode=str(spec.intervention_mode),
                configuration_sha256=_digest(
                    {"alpha": 0.0, "kind": "original-qwen-identity"}
                ),
            ),
            observation_status="recorded",
            observed_semantic_label=spec.semantic_label,
            hidden_sha256=_digest({"hidden": spec.sha256}),
            activation_sha256=_digest({"activation": spec.sha256}),
            logits_sha256=_digest({"logits": spec.sha256}),
            state_sha256=_digest({"state": spec.sha256}),
            access_trace_sha256=_digest({"trace": spec.sha256}),
            evidence_sha256=_digest({"evidence": spec.sha256}),
            weight_rail_revision=_WEIGHT_REVISION,
            atlas_head_revision=atlas_head_revision,
            numeric_summaries=(
                NumericSummary(
                    metric="activation_rms",
                    count=2,
                    total=2.0 * signal,
                    total_squares=2.0 * signal * signal,
                    minimum=signal,
                    maximum=signal,
                ),
            ),
            placebo_effects=(),
            runtime=_RUNTIME_PROVENANCE,
        )
        return _FakeResult(measurement)


def _runtime_factory(body):
    pin = ModelPin.from_document(body["model_pin"])
    return cartography.CartographyRuntime(
        model=_FakeModel(pin), tensor_plans=(object(),)
    )


def _pin() -> ModelPin:
    return ModelPin(
        repo_id="local:qwen-cartography-fake",
        revision="fixture",
        bundle_fingerprint="b" * 64,
        bundle_manifest_sha256="c" * 64,
        code_revision=CODE_REVISION,
    )


def _jobs() -> list[dict[str, object]]:
    rows = []
    for layer in (0, 1):
        module = f"model.language_model.layers.{layer}.mlp.gate_proj"
        rows.append(
            {
                "coordinate": {
                    "layer": layer,
                    "module": module,
                    "row_end": 2,
                    "row_start": 0,
                    "tensor": f"{module}.weight",
                },
                "intervention_mode": "passive",
                "probe_family": "arithmetic",
                "start_layer": 0,
                "stop_layer": layer + 1,
            }
        )
    return rows


class Qwen38O1CartographyTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAtlas.stores.clear()
        _FakeAtlas.histories.clear()
        _FakeProbe.calls.clear()
        _AuthenticProbe.calls.clear()
        _FakeProbe.failures_remaining = 0
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen-o1-loop-test-", dir=Path.cwd()
        )
        self.base = Path(self.temporary.name)
        self.bundle = self.base / "bundle"
        self.bundle.mkdir()
        self.tokens = (1, 4, 9)
        self.prompt_sha = prompt_token_sha256(self.tokens)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prepare(self, root: Path):
        return cartography.prepare_manifest(
            root,
            bundle_root=self.bundle,
            model_pin=_pin(),
            prompt_token_ids=self.tokens,
            prompt_sha256=self.prompt_sha,
            jobs=_jobs(),
            code_revision=CODE_REVISION,
            seed=17,
        )

    def _run(self, root: Path, **changes):
        options = {
            "max_jobs": 0,
            "max_seconds": 10.0,
            "runtime_factory": _runtime_factory,
            "probe_factory": _FakeProbe,
            "atlas_factory": _FakeAtlas,
            "stream_factory": _FakeStream,
        }
        options.update(changes)
        return cartography.run_cartography(root, **options)

    def test_prepare_is_deterministic_atomic_and_non_overwriting(self) -> None:
        first_root = self.base / "first"
        second_root = self.base / "second"
        first = self._prepare(first_root)
        second = self._prepare(second_root)
        self.assertEqual(first, second)
        manifest = first_root / cartography.MANIFEST_NAME
        self.assertEqual(
            manifest.read_bytes(),
            json.dumps(
                first,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
        )
        with self.assertRaisesRegex(
            cartography.O1CartographyCliError, "already exists"
        ):
            self._prepare(first_root)

    def test_bundle_pin_is_the_sealed_manifest_identity_not_raw_file_hash(
        self,
    ) -> None:
        body = {
            "checkpoint_complete": True,
            "layout_fingerprint": "b" * 64,
            "logical_model": {
                "repo_id": "local:qwen-cartography-fake",
                "revision": "fixture",
            },
        }
        seal = _digest(body)
        document = {
            "body": body,
            "schema": cartography.QWEN38_BUNDLE_SCHEMA,
            "sha256": seal,
        }
        raw = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        (self.bundle / "bundle.json").write_bytes(raw)
        pin = ModelPin(
            repo_id="local:qwen-cartography-fake",
            revision="fixture",
            bundle_fingerprint="b" * 64,
            bundle_manifest_sha256=seal,
            code_revision=CODE_REVISION,
        )
        manifest_body = {
            "bundle": {"manifest_sha256": seal, "root": str(self.bundle)},
            "model_pin": pin.to_document(),
        }
        self.assertEqual(cartography._bundle_document(manifest_body), document)
        manifest_body["bundle"]["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(cartography.O1CartographyCliError, "manifest pin"):
            cartography._bundle_document(manifest_body)

    def test_complete_frontier_runs_exactly_twice_then_reopens_and_queries(
        self,
    ) -> None:
        run_root = self.base / "run"
        self._prepare(run_root)
        first = self._run(run_root)
        self.assertEqual(first["attempts_executed"], 2)
        self.assertEqual(first["coverage"]["promoted_jobs"], 2)
        self.assertTrue(first["coverage"]["complete"])
        self.assertEqual(len(_FakeProbe.calls), 2)
        self.assertEqual(
            first["attached_atlas_receipt_sha256s"],
            first["atlas"]["measurement_sha256s"],
        )

        reopened = self._run(run_root)
        self.assertEqual(reopened["attempts_executed"], 0)
        self.assertEqual(len(_FakeProbe.calls), 2)
        status = cartography.status_cartography(
            run_root,
            runtime_factory=_runtime_factory,
            atlas_factory=_FakeAtlas,
            stream_factory=_FakeStream,
        )
        self.assertEqual(status["coverage"]["promoted_jobs"], 2)
        queried = cartography.query_cartography(
            run_root,
            prompt_sha256=self.prompt_sha,
            runtime_factory=_runtime_factory,
            atlas_factory=_FakeAtlas,
        )
        self.assertEqual(len(queried["measurements"]), 2)
        absent = cartography.query_cartography(
            run_root,
            prompt_sha256="f" * 64,
            runtime_factory=_runtime_factory,
            atlas_factory=_FakeAtlas,
        )
        self.assertEqual(absent["measurements"], [])
        manifest_body = cartography._load_manifest(run_root)[1]
        scheduler = cartography.O1Cartographer.restore(
            run_root / cartography.SCHEDULER_NAME,
            code_pin=CODE_REVISION,
            model_pin=manifest_body["model_pin_sha256"],
            stream=_FakeStream(),
        )
        self.assertEqual([outcome.read_bytes for outcome in scheduler.outcomes], [7, 7])

    def test_multi_prompt_registry_runs_cross_product_and_queries_separately(
        self,
    ) -> None:
        run_root = self.base / "multi-prompt"
        other_tokens = (2, 5, 8, 13, 21)
        other_sha = prompt_token_sha256(other_tokens)
        manifest = cartography.prepare_manifest(
            run_root,
            bundle_root=self.bundle,
            model_pin=_pin(),
            prompts=(
                {"sha256": self.prompt_sha, "token_ids": list(self.tokens)},
                {"sha256": other_sha, "token_ids": list(other_tokens)},
            ),
            jobs=_jobs(),
            code_revision=CODE_REVISION,
            seed=17,
        )
        self.assertEqual(len(manifest["body"]["prompts"]), 2)
        self.assertEqual(len(manifest["body"]["jobs"]), 4)

        report = self._run(run_root)
        self.assertEqual(report["attempts_executed"], 4)
        self.assertEqual(report["coverage"]["promoted_jobs"], 4)
        first = cartography.query_cartography(
            run_root,
            prompt_sha256=self.prompt_sha,
            runtime_factory=_runtime_factory,
            atlas_factory=_FakeAtlas,
        )
        second = cartography.query_cartography(
            run_root,
            prompt_sha256=other_sha,
            runtime_factory=_runtime_factory,
            atlas_factory=_FakeAtlas,
        )
        self.assertEqual(len(first["measurements"]), 2)
        self.assertEqual(len(second["measurements"]), 2)
        self.assertTrue(
            {row["sha256"] for row in first["measurements"]}.isdisjoint(
                row["sha256"] for row in second["measurements"]
            )
        )

        document = json.loads((run_root / cartography.MANIFEST_NAME).read_text())
        document["body"]["prompts"][0]["token_ids"][0] += 1
        document["sha256"] = _digest(document["body"])
        (run_root / cartography.MANIFEST_NAME).write_bytes(
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        with self.assertRaisesRegex(
            cartography.O1CartographyCliError, "prompt token IDs"
        ):
            cartography.status_cartography(run_root)

    def test_external_prompt_label_reaches_atlas_label_query(self) -> None:
        run_root = self.base / "external-label"
        label = "arithmetic/subtraction"
        label_source = _digest({"source": "fertig-exact"})
        label_evidence = _digest({"proof": "fraction-rref"})
        manifest = cartography.prepare_manifest(
            run_root,
            bundle_root=self.bundle,
            model_pin=_pin(),
            prompts=(
                {
                    "sha256": self.prompt_sha,
                    "token_ids": list(self.tokens),
                    "spec_defaults": {
                        "label_evidence_sha256": label_evidence,
                        "label_source_sha256": label_source,
                        "semantic_label": label,
                    },
                },
            ),
            jobs=(_jobs()[0],),
            code_revision=CODE_REVISION,
            seed=17,
        )
        spec = manifest["body"]["jobs"][0]["spec"]
        self.assertEqual(spec["semantic_label"], label)
        self.assertEqual(spec["label_evidence_sha256"], label_evidence)

        report = self._run(run_root)
        self.assertEqual(report["coverage"]["promoted_jobs"], 1)
        queried = cartography.query_cartography(
            run_root,
            semantic_label=label,
            runtime_factory=_runtime_factory,
            atlas_factory=_FakeAtlas,
        )
        self.assertEqual(len(queried["measurements"]), 1)
        self.assertEqual(queried["measurements"][0]["semantic_label"], label)

    def test_crash_after_outcome_checkpoint_reuses_atlas_proof(self) -> None:
        run_root = self.base / "crash"
        self._prepare(run_root)
        original = cartography.O1Cartographer.attach_atlas_receipt
        crashed = False

        def crash_once(scheduler, **kwargs):
            nonlocal crashed
            if not crashed:
                crashed = True
                raise OSError("simulated crash before receipt attachment")
            return original(scheduler, **kwargs)

        with mock.patch.object(
            cartography.O1Cartographer,
            "attach_atlas_receipt",
            autospec=True,
            side_effect=crash_once,
        ):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                self._run(run_root, max_jobs=1)
        self.assertEqual(len(_FakeProbe.calls), 1)

        resumed = self._run(run_root)
        self.assertEqual(resumed["receipts_reconciled"], 1)
        self.assertEqual(len(_FakeProbe.calls), 2)
        self.assertEqual(resumed["coverage"]["promoted_jobs"], 2)

    def test_crash_reconciliation_ingests_authentic_outcome_exactly_once(self) -> None:
        run_root = self.base / "authentic-crash"
        cartography.prepare_manifest(
            run_root,
            bundle_root=self.bundle,
            model_pin=_pin(),
            prompt_token_ids=self.tokens,
            prompt_sha256=self.prompt_sha,
            jobs=_jobs()[:1],
            code_revision=CODE_REVISION,
            seed=17,
        )
        original = cartography.O1Cartographer.attach_atlas_receipt
        crashed = False

        def crash_once(scheduler, **kwargs):
            nonlocal crashed
            if not crashed:
                crashed = True
                raise OSError("simulated crash before authentic attachment")
            return original(scheduler, **kwargs)

        with mock.patch.object(
            cartography.O1Cartographer,
            "attach_atlas_receipt",
            autospec=True,
            side_effect=crash_once,
        ):
            with self.assertRaisesRegex(OSError, "authentic attachment"):
                self._run(
                    run_root,
                    max_jobs=1,
                    probe_factory=_AuthenticProbe,
                )
        self.assertEqual(len(_AuthenticProbe.calls), 1)

        resumed = self._run(
            run_root,
            probe_factory=_AuthenticProbe,
        )
        self.assertEqual(resumed["receipts_reconciled"], 1)
        self.assertEqual(resumed["attempts_executed"], 0)
        self.assertEqual(len(resumed["ooe"]["learning_receipts"]), 1)
        self.assertEqual(resumed["ooe"]["controller_last_temporal_index"], 0)
        self.assertEqual(len(_AuthenticProbe.calls), 1)

        reopened = self._run(
            run_root,
            probe_factory=_AuthenticProbe,
        )
        self.assertEqual(reopened["ooe"]["learning_receipts"], [])
        self.assertEqual(reopened["ooe"]["controller_last_temporal_index"], 0)

    def test_failed_probe_is_durable_and_retryable_on_reopen(self) -> None:
        run_root = self.base / "failure"
        self._prepare(run_root)
        _FakeProbe.failures_remaining = 1
        failed = self._run(run_root, max_jobs=1)
        self.assertEqual(failed["coverage"]["retryable_jobs"], 1)
        self.assertEqual(failed["coverage"]["succeeded_jobs"], 0)
        resumed = self._run(run_root, max_jobs=1)
        self.assertEqual(resumed["coverage"]["succeeded_jobs"], 1)
        self.assertEqual(resumed["coverage"]["attempts"], 2)

    def test_default_o1_organism_sidecar_survives_fresh_status_restore(self) -> None:
        run_root = self.base / "organism"
        self._prepare(run_root)
        report = self._run(run_root, max_jobs=1, stream_factory=None)
        self.assertEqual(report["coverage"]["succeeded_jobs"], 1)

        state = json.loads((run_root / cartography.SCHEDULER_NAME).read_text())
        stream_state = state["stream_state"]
        self.assertGreaterEqual(state["outcomes"][0]["learning_progress"], 1.0)
        receipt = stream_state["sidecar"]
        self.assertEqual(receipt["schema"], "immer.o1-state-sidecar-receipt/v1")
        self.assertGreater(stream_state["tokens"], 0)
        self.assertIsNotNone(stream_state["tail"])
        sidecars = tuple(run_root.glob("o1-state.*.pt"))
        self.assertEqual([path.name for path in sidecars], [receipt["name"]])

        status = cartography.status_cartography(
            run_root,
            runtime_factory=_runtime_factory,
            atlas_factory=_FakeAtlas,
            stream_factory=None,
        )
        self.assertEqual(status["coverage"]["succeeded_jobs"], 1)
        self.assertEqual(tuple(run_root.glob("o1-state.*.pt")), sidecars)

    def test_authentic_o1_measurements_resume_into_atomic_ooe_crystals(self) -> None:
        run_root = self.base / "authentic-ooe"
        ooe_root = self.base / "custom-ooe-store"
        self._prepare(run_root)

        first = self._run(
            run_root,
            max_jobs=1,
            probe_factory=_AuthenticProbe,
            ooe_root=ooe_root,
        )
        self.assertEqual(first["ooe"]["qwen_probe_calls"], 1)
        self.assertEqual(first["ooe"]["reused_atlas_proofs"], 0)
        self.assertEqual(len(first["ooe"]["learning_receipts"]), 1)
        self.assertEqual(len(first["ooe"]["promotions"]), 1)
        self.assertEqual(first["ooe"]["saved_qwen_forwards"], 0)
        first_learning = first["ooe"]["learning_receipts"][0]
        self.assertEqual(
            first_learning["learning_receipt"]["source_action"],
            "qwen_fallback",
        )
        self.assertEqual(
            first_learning["learning_receipt"]["target_action"],
            "probe_coordinate",
        )
        self.assertEqual(first["ooe"]["root"], str(ooe_root.absolute()))

        second = self._run(
            run_root,
            max_jobs=1,
            probe_factory=_AuthenticProbe,
            ooe_root=ooe_root,
        )
        self.assertEqual(second["ooe"]["qwen_probe_calls"], 1)
        self.assertEqual(len(second["ooe"]["learning_receipts"]), 1)
        self.assertEqual(
            second["ooe"]["learning_receipts"][0]["learning_receipt"][
                "source_action"
            ],
            "probe_coordinate",
        )
        self.assertEqual(second["ooe"]["controller_last_temporal_index"], 1)
        self.assertEqual(len(second["ooe"]["site_identity_sha256s"]), 2)

        reopened = self._run(
            run_root,
            probe_factory=_AuthenticProbe,
            ooe_root=ooe_root,
        )
        self.assertEqual(reopened["attempts_executed"], 0)
        self.assertEqual(reopened["ooe"]["learning_receipts"], [])
        self.assertEqual(reopened["ooe"]["qwen_probe_calls"], 0)
        self.assertFalse(reopened["ooe"]["controller_snapshot"]["changed"])
        self.assertEqual(len(_AuthenticProbe.calls), 2)

    def test_reused_atlas_proof_learns_without_probe_or_forward_savings(self) -> None:
        run_root = self.base / "ooe-reuse"
        manifest = self._prepare(run_root)
        atlas_path = run_root / cartography.ATLAS_NAME
        atlas_path.mkdir()
        atlas = _FakeAtlas(
            atlas_path,
            model_pin=_pin(),
            tensor_plans=(object(),),
        )
        for row in manifest["body"]["jobs"]:
            spec = cartography._spec_from_record(row["spec"])
            precomputed = _AuthenticProbe(_FakeModel(_pin())).execute(
                spec,
                atlas_head_revision=atlas.revision(),
            ).measurement
            atlas.append_measurement(precomputed)
        _AuthenticProbe.calls.clear()

        report = self._run(
            run_root,
            max_jobs=1,
            probe_factory=_AuthenticProbe,
        )
        self.assertEqual(report["ooe"]["qwen_probe_calls"], 0)
        self.assertEqual(report["ooe"]["reused_atlas_proofs"], 1)
        self.assertEqual(len(report["ooe"]["learning_receipts"]), 1)
        self.assertTrue(
            report["ooe"]["learning_receipts"][0]["reused_atlas_proof"]
        )
        self.assertEqual(report["ooe"]["saved_qwen_forwards"], 0)
        self.assertEqual(_AuthenticProbe.calls, [])

    def test_ooe_atlas_verifier_rejects_forged_historical_event(self) -> None:
        run_root = self.base / "ooe-historical-revision"
        manifest = self._prepare(run_root)
        atlas_path = run_root / cartography.ATLAS_NAME
        atlas_path.mkdir()
        atlas = _FakeAtlas(
            atlas_path,
            model_pin=_pin(),
            tensor_plans=(object(),),
        )
        measurements = []
        for row in manifest["body"]["jobs"]:
            spec = cartography._spec_from_record(row["spec"])
            measurement = _AuthenticProbe(_FakeModel(_pin())).execute(
                spec,
                atlas_head_revision=atlas.revision(),
            ).measurement
            atlas.append_measurement(measurement)
            measurements.append(measurement)

        verifier = cartography._atlas_revision_verifier(atlas, measurements)
        historical = measurements[0].atlas_head_revision
        self.assertLess(historical.sequence, atlas.revision().sequence)
        self.assertTrue(verifier(historical))
        forged = GraphRevision(
            historical.sequence,
            _digest({"forged-event": historical.sequence}),
        )
        self.assertFalse(verifier(forged))

    def test_ooe_atlas_measurement_auth_rejects_concurrent_head_change(self) -> None:
        run_root = self.base / "ooe-concurrent-atlas-head"
        manifest = self._prepare(run_root)
        atlas_path = run_root / cartography.ATLAS_NAME
        atlas_path.mkdir()
        atlas = _FakeAtlas(atlas_path, model_pin=_pin(), tensor_plans=(object(),))
        spec = cartography._spec_from_record(manifest["body"]["jobs"][0]["spec"])
        measurement = _AuthenticProbe(_FakeModel(_pin())).execute(
            spec,
            atlas_head_revision=atlas.revision(),
        ).measurement
        atlas.append_measurement(measurement)
        original_query = atlas.query_by_prompt_signature

        def moving_query(prompt_sha256):
            result = original_query(prompt_sha256)
            atlas.history.append(
                GraphRevision(
                    atlas.history[-1].sequence + 1,
                    _digest({"concurrent": atlas.history[-1].sequence + 1}),
                )
            )
            return result

        atlas.query_by_prompt_signature = moving_query
        verifier = cartography._AtlasRevisionMembership(atlas)
        with self.assertRaisesRegex(
            cartography.O1CartographyCliError,
            "head changed during",
        ):
            verifier.authenticate_measurement(measurement)

    def test_ooe_snapshot_tamper_is_rejected_on_exact_resume(self) -> None:
        run_root = self.base / "ooe-tamper"
        self._prepare(run_root)
        first = self._run(
            run_root,
            probe_factory=_AuthenticProbe,
        )
        snapshot_sha256 = first["ooe"]["controller_snapshot"]["payload_sha256"]
        clean_resume = self._run(
            run_root,
            max_jobs=0,
            probe_factory=_AuthenticProbe,
        )
        self.assertEqual(
            clean_resume["ooe"]["controller_snapshot"]["payload_sha256"],
            snapshot_sha256,
        )
        state_name = hashlib.sha256(
            cartography.CONTROLLER_STATE_NAME.encode("utf-8")
        ).hexdigest()
        state_path = (
            run_root / cartography.OOE_NAME / "state" / f"{state_name}.state"
        )
        envelope = json.loads(state_path.read_text())
        envelope["payload_sha256"] = "0" * 64
        state_path.chmod(0o600)
        state_path.write_bytes(
            json.dumps(
                envelope,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        with self.assertRaises(CrystalTamperError):
            self._run(
                run_root,
                max_jobs=0,
                probe_factory=_AuthenticProbe,
            )

    def test_crash_between_crystal_promotion_and_snapshot_recovers_exactly(
        self,
    ) -> None:
        run_root = self.base / "ooe-promotion-crash"
        self._prepare(run_root)
        original = cartography.OoeController.save_snapshot
        calls = 0

        def crash_second_save(controller, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated crash after Crystal promotion")
            return original(controller, **kwargs)

        with mock.patch.object(
            cartography.OoeController,
            "save_snapshot",
            autospec=True,
            side_effect=crash_second_save,
        ):
            with self.assertRaisesRegex(OSError, "after Crystal promotion"):
                self._run(
                    run_root,
                    probe_factory=_AuthenticProbe,
                )
        self.assertEqual(len(_AuthenticProbe.calls), 2)
        transaction = cartography._load_promotion_transaction(
            cartography.CrystalStore(run_root / cartography.OOE_NAME)
        )
        self.assertIsNotNone(transaction)
        self.assertEqual(transaction[0]["body"]["phase"], "prepared")

        resumed = self._run(
            run_root,
            probe_factory=_AuthenticProbe,
        )
        recovery = resumed["ooe"]["recovery_receipt"]
        self.assertIsNotNone(recovery)
        self.assertGreater(
            recovery["new_manifest_generation"],
            recovery["old_manifest_generation"],
        )
        self.assertEqual(len(recovery["recovered_site_sha256s"]), 2)
        self.assertEqual(resumed["ooe"]["learning_receipts"], [])
        self.assertEqual(resumed["ooe"]["qwen_probe_calls"], 0)
        self.assertEqual(len(_AuthenticProbe.calls), 2)
        committed = cartography._load_promotion_transaction(
            cartography.CrystalStore(run_root / cartography.OOE_NAME)
        )
        self.assertEqual(committed[0]["body"]["phase"], "committed")

        reopened = self._run(
            run_root,
            probe_factory=_AuthenticProbe,
        )
        self.assertIsNone(reopened["ooe"]["recovery_receipt"])
        self.assertFalse(reopened["ooe"]["controller_snapshot"]["changed"])

    def test_promotion_transaction_tamper_cannot_authorize_recovery(self) -> None:
        run_root = self.base / "ooe-transaction-tamper"
        self._prepare(run_root)
        self._run(run_root, probe_factory=_AuthenticProbe)
        state_name = hashlib.sha256(
            cartography.OOE_PROMOTION_STATE_NAME.encode("utf-8")
        ).hexdigest()
        state_path = (
            run_root / cartography.OOE_NAME / "state" / f"{state_name}.state"
        )
        envelope = json.loads(state_path.read_text())
        envelope["payload_sha256"] = "0" * 64
        state_path.chmod(0o600)
        state_path.write_bytes(
            json.dumps(
                envelope,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        with self.assertRaises(CrystalTamperError):
            self._run(run_root, probe_factory=_AuthenticProbe)

    def test_stream_factory_type_error_is_not_retried_or_weakened(self) -> None:
        calls = []

        def broken_factory(*, seed, sidecar):
            calls.append((seed, sidecar))
            raise TypeError("constructor body failed")

        with self.assertRaisesRegex(TypeError, "constructor body failed"):
            cartography._stream(
                broken_factory,
                seed=7,
                sidecar=self.base / cartography.O1_STATE_NAME,
            )
        self.assertEqual(len(calls), 1)

    def test_resealed_output_escape_and_state_symlink_are_rejected(self) -> None:
        run_root = self.base / "containment"
        document = self._prepare(run_root)
        document["body"]["outputs"]["scheduler"] = "../escaped.json"
        document["sha256"] = _digest(document["body"])
        (run_root / cartography.MANIFEST_NAME).write_bytes(
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        with self.assertRaisesRegex(cartography.O1CartographyCliError, "output"):
            cartography.status_cartography(run_root)

        second = self.base / "symlink"
        self._prepare(second)
        outside = self.base / "outside-state.json"
        outside.write_text("untouched", encoding="utf-8")
        (second / cartography.SCHEDULER_NAME).symlink_to(outside)
        with self.assertRaises(Exception):
            self._run(second, max_jobs=1)
        self.assertEqual(outside.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
