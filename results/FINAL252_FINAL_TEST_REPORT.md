# Final252 final independent-test report

## Locked evaluation state

- Final-test role: `pristine_final_test` (18 trajectories).
- The separate `historical_audit_test` role was not included.
- Final training size: 252 trajectories.
- Five locked checkpoints: seeds 0, 1, 2, 3, and 4.
- Model parameter count: 7,984,355.
- No retraining, fine-tuning, checkpoint reselection, hyperparameter change, loss change, or post-test model modification was performed.
- Final-test truth was accessed only after training size, architecture, hyperparameters, checkpoints, evaluation metrics and spatial-information control had been locked.

## Validation model-selection performance (locked; not recomputed)

| Metric | Five-seed mean ± sample SD |
|---|---:|
| Delta-phase relative L2 | 0.0461006 ± 0.0102263 |
| Interface MAE (nm) | 4.48013 ± 1.44883 |
| Maximum-depth MAE (nm) | 6.86713 ± 2.66121 |

## Final independent-test performance

Five-seed summaries average state-level metrics within each time group for each seed, then report mean ± sample SD across the five independent training seeds.

| Time group | States | Delta-phase relative L2 | Interface MAE (nm) | Maximum-depth MAE (nm) |
|---|---:|---:|---:|---:|
| all nonzero | 468 | 0.0464563231 ± 0.01323240949 | 7.292568182 ± 2.191978951 | 12.23399178 ± 5.110611834 |
| training grid | 216 | 0.04785848082 ± 0.01218407043 | 5.309640296 ± 1.538913714 | 8.987954101 ± 3.74422467 |
| withheld interpolation | 126 | 0.04520483798 ± 0.01304927716 | 6.615016436 ± 1.981319612 | 10.99552077 ± 4.731444549 |
| further time probes | 126 | 0.04530410928 ± 0.01522332552 | 11.36942488 ± 3.522551933 | 19.03709881 ± 7.842532273 |

## Ensemble mean-field performance

| Time group | States | Delta-phase relative L2 | Interface MAE (nm) | Maximum-depth MAE (nm) |
|---|---:|---:|---:|---:|
| all nonzero | 468 | 0.03634834017 | 6.002771452 | 10.66413128 |
| training grid | 216 | 0.03792074601 | 4.380682502 | 7.834160923 |
| withheld interpolation | 126 | 0.03509048579 | 5.458381133 | 9.387294141 |
| further time probes | 126 | 0.03491064168 | 9.327885686 | 16.79234618 |

## Initial condition and integrity checks

- Maximum absolute predicted delta phase at t = 0 across all trajectories and seeds: 0.
- Single-seed state metrics finite: YES.
- Ensemble state metrics finite: YES.
- Evaluated nonzero states per seed: 468.
- Auxiliary scalar heads were not assigned new post hoc metrics; the locked three spatial metrics remain the formal final-test endpoints.

## Predefined representative-case rule

Trajectories were ranked by ensemble all-26-time delta-phase relative L2. No visual case selection was used.

- Lower-quartile-error trajectory: `new108_fb12_network_12`
- Median-error trajectory: `new108_fb12_moderate_12`
- Upper-quartile-error trajectory: `new108_fb05_moderate_12`

## Status

**FINAL_TEST_STATUS = COMPLETE**

- Evaluation wall time: 116.4 s
- Post-test model modification: NONE
