# Prediction errors against the saved boundary bracket (TA request 4)

This is an interim evidence-based comparison using the saved **default-only** bracket from `artifacts/experiments/2026-10-21-boundary/verification.json`: last known success **114,688** and first typed CUDA OOM **131,072**, so under monotonicity the unknown maximum is an integer in **[114,688, 131,071]**. No midpoint is treated as observed. Signed error is `(prediction − measured limit) / measured limit × 100`.

## Table A — committed predictions (original prediction, unchanged)

The three scenario values below are the committed uncapped limits. The optimistic config-capped value is also shown; it is capped at 131,072 by the model config. Each error column evaluates the prediction at one bracket endpoint; the true error is somewhere in the corresponding endpoint interval, not one exact number.

| Committed prediction | Tokens | Error vs 114,688 lower bound | Error vs 131,071 upper bound |
|---|---:|---:|---:|
| Conservative (uncapped) | 76,024 | -33.712333% | -41.997848% |
| Central (uncapped) | 111,232 | -3.013393% | -15.136071% |
| Optimistic (uncapped) | 146,740 | +27.947126% | +11.954589% |
| Optimistic (config-capped) | 131,072 | +14.285714% | +0.000763% |

## Table B — post-measurement fitted peak model (separate from Table A)

The fits below use OLS on per-length medians from the same GPU study data (`data/processed/p1-2026-10-19/analysis.json`; 9 lengths; fit method explicitly claims no statistical CI). They are in-sample fitted models, not advance predictions. A fitted peak line is converted to a candidate crossing with `floor((usable_bytes − intercept) / slope)` using the original usable budget, 15,527,116,800 bytes. The default fit can be compared with the default OOM bracket; the expandable fit cannot be assigned a measured boundary error because no expandable OOM was recorded.

| Allocator fit | Slope (B/token) | Intercept (B) | Fitted crossing (tokens) | Error vs 114,688 lower bound | Error vs 131,071 upper bound | Data status |
|---|---:|---:|---:|---:|---:|---|
| default | 95236.94686064 | 3097750041.813 | 130,509 | +13.794817% | -0.428775% | Same-allocator bracket; fit uses same study GPU data |
| expandable | 95248.00000000 | 3095948800.000 | 130,513 | MISSING | MISSING | No expandable OOM endpoint; comparison to this bracket is not valid |

The default fitted crossing is 130,509 tokens, but it is not a measured boundary; the default OOM occurs at 131,072 and is attributed in the saved inspection to a framework mask-allocation cliff. The fit therefore does not establish smooth memory-pressure behavior at the failure point. Recompute both tables after the paired bisection; do not reuse the default interval for expandable.
