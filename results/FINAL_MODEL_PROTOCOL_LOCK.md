# Final model protocol lock

## Decision

- Final training size: **252 trajectories**
- Fixed validation size: **36 trajectories**
- Independent evaluation roles: **18 + 18 trajectories**
- The 36 trajectories in the 288-candidate pool but outside the 252 nested subset remain training candidates only; they are not reassigned or deleted.
- No additional COMSOL data generation or training-size search is authorized by this protocol.

## Model and optimization protocol

- Architecture: continuous-time analytically constrained U-Net with two auxiliary scalar heads
- Parameters: 7,984,355 trainable parameters
- base_channels: 32
- batch_size: 2
- learning_rate: 2e-4
- interface_lambda: 2.0
- dynamic_weight_cap: 4.0
- gradient_weight: 0.03
- weight_decay: 1e-4
- Optimizer: AdamW
- AMP: enabled
- Gradient clipping: 2.0
- MAX_OPTIMIZER_STEPS: 45,360
- VALIDATION_INTERVAL_STEPS: 1,296
- EARLY_STOP_PATIENCE_STEPS: 7,776
- Minimum improvement: 1e-5
- Scheduler: CosineAnnealingLR, stepped after each successful optimizer update
- T_max: 45,360; eta_min: 0; warm-up: none

## Loss configuration

- Field loss: global L1/0.01 + 0.05 × capped relative L2 + 0.35 × dynamically weighted L1/0.01 + 0.03 × spatial-gradient L1/0.01.
- Relative-L2 cap: 20; dynamic-weight cap: 4.0; interface multiplier: 2.0.
- Scalar auxiliary loss: standardized smooth-L1 mass loss + standardized smooth-L1 galvanic-current loss.
- Total loss: field loss + 0.25 × scalar auxiliary loss.

## Time protocol

- Training grid (h): 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24
- t = 0: handled only by the analytic initial-condition constraint.
- Withheld interpolation times not used for model selection (h): 3, 5, 7, 9, 14, 18, 22
- Further independent time probes not used for model selection (h): 11, 13, 15, 17, 19, 21, 23

## Five-seed checkpoint audit

| Seed | Best step | Stop step | Early stop | delta-phase relative L2 | Interface MAE (nm) | Maximum-depth MAE (nm) | Best validation L_total | SHA-256 | Status |
|---:|---:|---:|:---:|---:|---:|---:|---:|:---|:---:|
| 0 | 42768 | 45360 | FALSE | 0.0358149040 | 3.099684 | 4.483153 | 0.0118325819 | `e6eec858705352d006880880b104481caa004d56883c6d88677e7a0d54930f5f` | PASS |
| 1 | 45360 | 45360 | FALSE | 0.0536395643 | 5.634718 | 8.024514 | 0.0199057434 | `84d3eb644e6033c0c1fc0ba23dd3ec8e123db49fa7ac28d7257144cfbcd08f54` | PASS |
| 2 | 45360 | 45360 | FALSE | 0.0387498875 | 3.697536 | 5.055501 | 0.0132676888 | `2e22049d8f7dc032ced2e40f5233ecfa3ac4fb71864b953c5bc053d4271acc29` | PASS |
| 3 | 44064 | 45360 | FALSE | 0.0424698789 | 3.563667 | 5.797040 | 0.0154530696 | `708222d478f3d680a62074616b69142e23a506aaaedc370d7c4f261a8703a066` | PASS |
| 4 | 38880 | 45360 | FALSE | 0.0598286245 | 6.405049 | 10.975448 | 0.0236136626 | `4786e90ce557753f6b6efc9d0e2eaf93a81ec0ae31a43426c292109512f72764` | PASS |

## Five-seed validation summary

- delta-phase relative L2: 0.0461006 ± 0.0102263
- Interface MAE: 4.48013 ± 1.44883 nm
- Maximum-depth MAE: 6.86713 ± 2.66121 nm
- Standard deviation: sample SD (ddof = 1).

## Data-role independence

- Role sizes: `{"train252": 252, "validation36": 36, "independent18_a": 18, "independent18_b": 18}`
- ID overlaps: `{"train252__validation36": 0, "train252__independent18_a": 0, "train252__independent18_b": 0, "validation36__independent18_a": 0, "validation36__independent18_b": 0, "independent18_a__independent18_b": 0}`
- Duplicate recorded microstructure hashes: 0
- Duplicate canonical smooth microstructures: 0
- Duplicate microstructure paths: 0
- Duplicate trajectory paths: 0
- Missing metadata/files: 0/0
- Final 252 training subset balance: 28 trajectories in each of the 9 beta-fraction × morphology cells.
- Excluded training candidates retained: 36
- Independent-evaluation target truth read during this audit: **NO**

## Paper-method conflict record

> The final formal training protocol must replace the previous 35-epoch description with the unified optimizer-step protocol. The Methods section will be updated before final-test evaluation.

## Final-test gate

**FINAL_TEST_GATE = READY**

No unresolved checkpoint, metric, data-role, or time-protocol anomaly was found.
