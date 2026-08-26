from __future__ import annotations

import hashlib
import base64
import json
import warnings
import unittest

import numpy as np

from immer.runtimes.ooe.consensus import complete_adjacency
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.math_core import array_sha256
from immer.runtimes.ooe.reservoir import (
    PSLiftedReservoirAgent,
    PSLiftedReservoirConfig,
    ReservoirIntegrityError,
)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _target(bit: float) -> np.ndarray:
    return np.asarray([bit < 0.0, bit > 0.0], dtype=np.float64)


def _fit_delay(
    agent: PSLiftedReservoirAgent,
    bits: np.ndarray,
    *,
    delay: int,
    prefix: str,
) -> None:
    agent.reset_state()
    for index, bit in enumerate(bits):
        value = np.asarray([bit], dtype=np.float64)
        if index < delay:
            agent.advance(value)
            continue
        agent.observe(
            value,
            _target(float(bits[index - delay])),
            evidence_sha256=_sha({"prefix": prefix, "index": index}),
            solve=False,
        )
    agent.solve_readout()


def _delay_accuracy(
    agent: PSLiftedReservoirAgent,
    bits: np.ndarray,
    *,
    delay: int,
) -> float:
    agent.reset_state()
    correct = 0
    total = 0
    for index, bit in enumerate(bits):
        prediction = agent.predict(np.asarray([bit], dtype=np.float64))
        if index < delay:
            continue
        expected = int(bits[index - delay] > 0.0)
        correct += int(prediction.selected == expected)
        total += 1
    return correct / total


class PSLiftedReservoirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PSLiftedReservoirConfig(
            input_size=1,
            output_size=2,
            nodes=16,
            seed=77,
            spectral_radius=0.95,
            input_scale=0.2,
            leak_rate=1.0,
            ridge=1e-3,
        )

    def test_fixed_core_is_deterministic_and_seed_bound(self) -> None:
        first = PSLiftedReservoirAgent(self.config)
        second = PSLiftedReservoirAgent(self.config)
        changed = PSLiftedReservoirAgent(
            PSLiftedReservoirConfig(
                input_size=1,
                output_size=2,
                nodes=16,
                seed=78,
            )
        )
        self.assertEqual(first.core_sha256, second.core_sha256)
        self.assertNotEqual(first.core_sha256, changed.core_sha256)
        sequence = ([1.0], [-1.0], [0.5], [0.25])
        first_states = [first.predict(value).state_sha256 for value in sequence]
        second_states = [second.predict(value).state_sha256 for value in sequence]
        self.assertEqual(first_states, second_states)

    def test_delayed_sequence_memory_beats_no_memory(self) -> None:
        config = PSLiftedReservoirConfig(
            input_size=1,
            output_size=2,
            nodes=32,
            seed=42,
            spectral_radius=0.95,
            input_scale=0.2,
            leak_rate=1.0,
            ridge=1e-3,
        )
        rng = np.random.default_rng(9)
        train = rng.choice(np.asarray([-1.0, 1.0]), size=3_000)
        test = rng.choice(np.asarray([-1.0, 1.0]), size=800)
        agent = PSLiftedReservoirAgent(config)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _fit_delay(agent, train, delay=7, prefix="delay-seven")
            accuracy = _delay_accuracy(agent, test, delay=7)
        current_bit_baseline = float(
            np.mean((test[7:] > 0.0) == (test[:-7] > 0.0))
        )
        self.assertGreater(accuracy, 0.84)
        self.assertGreater(accuracy - current_bit_baseline, 0.30)

    def test_distributed_sufficient_statistics_match_central_learning(self) -> None:
        config = PSLiftedReservoirConfig(
            input_size=1,
            output_size=2,
            nodes=12,
            seed=123,
            spectral_radius=0.9,
            input_scale=0.1,
            leak_rate=0.8,
        )
        rng = np.random.default_rng(555)
        sequences = [
            rng.choice(np.asarray([-1.0, 1.0]), size=300) for _ in range(8)
        ]
        replicas = [PSLiftedReservoirAgent(config) for _ in range(4)]
        central = PSLiftedReservoirAgent(config)
        for sequence_index, sequence in enumerate(sequences):
            replica = replicas[sequence_index % len(replicas)]
            replica.reset_state()
            central.reset_state()
            for index, bit in enumerate(sequence):
                value = np.asarray([bit], dtype=np.float64)
                if index < 3:
                    replica.advance(value)
                    central.advance(value)
                    continue
                target = _target(float(sequence[index - 3]))
                evidence = _sha(
                    {"sequence": sequence_index, "index": index}
                )
                replica.observe(
                    value,
                    target,
                    evidence_sha256=evidence,
                    solve=False,
                )
                central.observe(
                    value,
                    target,
                    evidence_sha256=evidence,
                    solve=False,
                )
        for replica in replicas:
            replica.solve_readout()
        central.solve_readout()
        fused = PSLiftedReservoirAgent.fuse_ps_lifted(
            replicas,
            complete_adjacency(4),
            tolerance=1e-9,
            max_rounds=1_024,
        )
        self.assertTrue(fused.receipt.consensus.converged)
        self.assertEqual(fused.agent.sample_count, central.sample_count)
        self.assertEqual(
            fused.agent.evidence_sha256s,
            central.evidence_sha256s,
        )
        self.assertLess(
            float(np.max(np.abs(fused.agent.readout - central.readout))),
            1e-5,
        )
        test = rng.choice(np.asarray([-1.0, 1.0]), size=400)
        self.assertGreater(_delay_accuracy(fused.agent, test, delay=3), 0.9)

    def test_duplicate_evidence_is_rejected_locally_and_across_replicas(self) -> None:
        digest = _sha({"same": 1})
        first = PSLiftedReservoirAgent(self.config)
        first.observe([1.0], [0.0, 1.0], evidence_sha256=digest)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            first.observe([-1.0], [1.0, 0.0], evidence_sha256=digest)
        second = PSLiftedReservoirAgent(self.config)
        second.observe([-1.0], [1.0, 0.0], evidence_sha256=digest)
        with self.assertRaisesRegex(ValueError, "duplicate evidence"):
            PSLiftedReservoirAgent.fuse_ps_lifted(
                (first, second), complete_adjacency(2)
            )

    def test_canonical_roundtrip_preserves_executable_state(self) -> None:
        agent = PSLiftedReservoirAgent(self.config)
        for index, bit in enumerate((1.0, -1.0, 1.0, 1.0, -1.0)):
            agent.observe(
                [bit],
                _target(bit),
                evidence_sha256=_sha({"roundtrip": index}),
                solve=False,
            )
        agent.solve_readout()
        agent.predict([0.25])
        data = agent.to_bytes()
        restored = PSLiftedReservoirAgent.from_bytes(data)
        self.assertEqual(restored.to_bytes(), data)
        self.assertEqual(restored.sha256, agent.sha256)
        self.assertEqual(restored.predict([-0.75]), agent.predict([-0.75]))

    def test_tampered_payload_core_and_statistics_fail_closed(self) -> None:
        agent = PSLiftedReservoirAgent(self.config)
        agent.observe(
            [1.0],
            [0.0, 1.0],
            evidence_sha256=_sha({"tamper": 1}),
        )
        root = json.loads(agent.to_bytes())
        root["evidence_mass"] = 99.0
        with self.assertRaises(ReservoirIntegrityError):
            PSLiftedReservoirAgent.from_bytes(canonical_json_bytes(root))

    def test_self_consistent_but_impossible_gram_matrix_is_rejected(self) -> None:
        agent = PSLiftedReservoirAgent(self.config)
        for index, bit in enumerate((1.0, -1.0, 1.0)):
            agent.observe(
                [bit],
                _target(bit),
                evidence_sha256=_sha({"gram": index}),
                solve=False,
            )
        agent.solve_readout()
        root = json.loads(agent.to_bytes())
        descriptor = root["arrays"]["xtx"]
        xtx = np.frombuffer(
            base64.b64decode(descriptor["data_base64"]),
            dtype="<f8",
        ).reshape(descriptor["shape"]).copy()
        xtx[0, 1] += 7.0
        data = np.asarray(xtx, dtype="<f8", order="C").tobytes(order="C")
        root["arrays"]["xtx"] = {
            "byte_count": len(data),
            "data_base64": base64.b64encode(data).decode("ascii"),
            "dtype": "<f8",
            "raw_sha256": hashlib.sha256(data).hexdigest(),
            "shape": list(xtx.shape),
        }
        # ``array_sha256`` includes tensor shape while raw_sha256 does not.
        xty = np.frombuffer(
            base64.b64decode(root["arrays"]["xty"]["data_base64"]),
            dtype="<f8",
        ).reshape(root["arrays"]["xty"]["shape"])
        statistics = _sha(
            {
                "evidence_mass": root["evidence_mass"],
                "evidence_sha256s": root["evidence_sha256s"],
                "sample_count": root["sample_count"],
                "xtx_sha256": array_sha256(xtx),
                "xty_sha256": array_sha256(xty),
            }
        )
        root["statistics_sha256"] = statistics
        root["reservoir_sha256"] = _sha(
            {
                "core_sha256": root["core_sha256"],
                "readout_sha256": root["readout_sha256"],
                "state_sha256": root["state_sha256"],
                "statistics_sha256": statistics,
            }
        )
        with self.assertRaisesRegex(ReservoirIntegrityError, "not symmetric"):
            PSLiftedReservoirAgent.from_bytes(canonical_json_bytes(root))

        root = json.loads(agent.to_bytes())
        encoded = root["arrays"]["input_weights"]["data_base64"]
        root["arrays"]["input_weights"]["data_base64"] = (
            ("A" if encoded[0] != "A" else "B") + encoded[1:]
        )
        with self.assertRaises(ReservoirIntegrityError):
            PSLiftedReservoirAgent.from_bytes(canonical_json_bytes(root))

    def test_retention_changes_only_learned_statistics(self) -> None:
        agent = PSLiftedReservoirAgent(self.config)
        agent.observe(
            [1.0],
            [0.0, 1.0],
            evidence_sha256=_sha({"retention": 1}),
        )
        core = agent.core_sha256
        mass = agent.evidence_mass
        agent.apply_retention(0.25)
        self.assertEqual(agent.core_sha256, core)
        self.assertAlmostEqual(agent.evidence_mass, mass * 0.25)


if __name__ == "__main__":
    unittest.main()
