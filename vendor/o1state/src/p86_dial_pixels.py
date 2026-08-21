"""P86 scoring: the dial law on pixels, from the three-arm visual telemetry.

Clause (a): last-fifth mean L1(dense)/mean L1(gated) >= 0.95 at overall
admission share <= 0.25.
Clause (b): per-decile instantaneous admission rate, deciles 3-10 all <= 0.25.
"""
import json

GATED = "results/visual_arm_a_residual_gated_metrics.jsonl"
DENSE = "results/visual_arm_b_residual_nogate_metrics.jsonl"
OUT = "results/visual_p86_dial.json"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


gated = load(GATED)
dense = load(DENSE)
assert len(gated) == len(dense)
n = len(gated)

fifth = n // 5
last_g = gated[4 * fifth:]
last_d = dense[4 * fifth:]
mean_l1_g = sum(r["l1"] for r in last_g) / len(last_g)
mean_l1_d = sum(r["l1"] for r in last_d) / len(last_d)
quality_ratio = mean_l1_d / mean_l1_g
admission_share = sum(r["gated"] for r in gated) / n

dec = n // 10
deciles = []
for i in range(10):
    rows = gated[i * dec: (i + 1) * dec if i < 9 else n]
    deciles.append(round(sum(r["gated"] for r in rows) / len(rows), 4))

result = {
    "registered": "P86",
    "n_rows": n,
    "clause_a": {
        "mean_l1_gated_last_fifth": round(mean_l1_g, 6),
        "mean_l1_dense_last_fifth": round(mean_l1_d, 6),
        "quality_ratio": round(quality_ratio, 4),
        "admission_share_overall": round(admission_share, 4),
        "bar_ratio_ge_095": quality_ratio >= 0.95,
        "bar_share_le_025": admission_share <= 0.25,
    },
    "clause_b": {
        "decile_admission_rates": deciles,
        "bar_deciles_3_10_le_025": all(d <= 0.25 for d in deciles[2:]),
        "settled_amplitude": deciles[-1],
        "language_reference": 0.25,
    },
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
