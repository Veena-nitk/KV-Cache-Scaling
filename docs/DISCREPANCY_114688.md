# Discrepancy at 114,688 tokens (Part A3)

The source is the preserved per-probe decomposition in `data/processed/p1-2026-10-19/discrepancy.csv`. This new table adds an explicit arithmetic **Unaccounted** residual and closure rows; it does not overwrite the source. The representative default probe is `5de4f220-242d-441c-985a-9cb69cfa2780` (three default repeats exist at this length); the expandable representative is `9a193cc4-9884-42b1-a6d3-176826691deb` (the saved cohort contains one repeat). Prediction is the central **tensor peak** 15,359,962,112 B = W + L(28,672 + 78,336); headroom is not added. `MiB = bytes / 1,048,576`. To preserve `discrepancy.csv`'s sign convention, gap is **measured − predicted**; negative means the measured peak is below the prediction.

| Allocator | Probe | Predicted tensor peak (B) | Measured peak (B) | Gap: measured − predicted (B) | Gap (MiB) | Unaccounted (B) | Unaccounted (MiB) | Closure (B) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | `5de4f220…` | 15,359,962,112 | 14,020,668,928 | -1,339,293,184 | -1,277.250000 | 0 | 0.000000 | 0 |
| expandable | `9a193cc4…` | 15,359,962,112 | 14,019,751,424 | -1,340,210,688 | -1,278.125000 | 0 | 0.000000 | 0 |

[`docs/DISCREPANCY_114688.csv`](DISCREPANCY_114688.csv) supplies every component row, including exact byte and MiB values, each allocator's **Unaccounted** row, the total gap, and a zero closure check. A zero arithmetic residual does **not** establish causal accounting: in particular, `transient_envelope_difference` is descriptive relative to an assumed activation envelope, not a measured activation-only term. The source marks unexplained storage categories unsupported even where their measured arithmetic value is zero. The difference between measured gaps is 917,504 B (0.875 MiB), default gap minus expandable gap.
