# W16 correction status

The source is repaired, but the old `96/96` number cannot be converted into a
valid end-to-end result from the archive alone. The archive does not contain the
27B donor NPZ, the living-host checkpoint, or the two host source modules.

The corrected runner now:

- computes the output from the **predicted** operand, never the true operand;
- reports the oracle emitter separately;
- fits the affine crystal on training cells only;
- replaces map and ordinal readout atomically;
- evaluates the nine unique task cells instead of calling repeated draws
  "fresh cells";
- runs multiple model and held-split seeds;
- records that the bridge uses the known inverse formula and is therefore a
  structured bridge, not an unqualified zero-label experiment.

Run in the original environment:

```bash
PYTHONPATH=/path/to/w16-semigroup-fixed \
python /path/to/w16-semigroup-fixed/w16/w16_twostep_organ_fixed.py \
  --o1-root /Users/bhkmie/Documents/Forschung/O1_juli \
  --donor-npz /path/to/donor_targets_twostep_pos_27b.npz \
  --output /path/to/w16_twostep_fixed.json
```

The legitimate headline after rerun is `aggregate.end_to_end_held_unique_mean`.
`aggregate.oracle_emitter_all_unique_mean` measures only the output mechanism.
