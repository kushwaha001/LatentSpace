# Experiment 07: strict vs lenient argument matching (held-out test, 108 tasks x 5)

Rules: `src/transit_agent/lenient_match.py`. Only the tool-call match changes; validity, permission and the confirmation check are unchanged.

| Model | success strict | success lenient | exact strict | exact lenient | F1 strict | F1 lenient | pass^5 strict | pass^5 lenient |
|---|---|---|---|---|---|---|---|---|
| base | 0.161 | **0.230** | 0.167 | 0.243 | 0.199 | 0.283 | 0.056 | 0.102 |
| distill | 0.170 | **0.215** | 0.200 | 0.269 | 0.230 | 0.308 | 0.102 | 0.120 |
| run_A | 0.239 | **0.322** | 0.241 | 0.328 | 0.272 | 0.371 | 0.194 | 0.259 |
| run_B | 0.202 | **0.272** | 0.228 | 0.306 | 0.259 | 0.350 | 0.148 | 0.194 |

## Success per language (strict → lenient)

| Model | en | hi | mr | hinglish |
|---|---|---|---|---|
| base | 0.18 → 0.25 | 0.14 → 0.21 | 0.14 → 0.21 | 0.19 → 0.24 |
| distill | 0.19 → 0.24 | 0.17 → 0.22 | 0.19 → 0.21 | 0.13 → 0.18 |
| run_A | 0.25 → 0.33 | 0.28 → 0.39 | 0.24 → 0.33 | 0.19 → 0.25 |
| run_B | 0.24 → 0.30 | 0.20 → 0.27 | 0.19 → 0.27 | 0.19 → 0.26 |

## Success per category (strict → lenient)

| Category | base | distill | run_A | run_B |
|---|---|---|---|---|
| declined | 0.23 → 0.23 | 0.53 → 0.53 | 0.88 → 0.88 | 0.65 → 0.65 |
| denied_tool | 0.07 → 0.07 | 0.00 → 0.00 | 0.07 → 0.07 | 0.07 → 0.07 |
| irreversible_confirmed | 0.30 → 0.30 | 0.03 → 0.03 | 0.00 → 0.00 | 0.06 → 0.06 |
| irreversible_unconfirmed | 0.03 → 0.03 | 0.00 → 0.00 | 0.10 → 0.10 | 0.00 → 0.00 |
| missing_argument | 0.05 → 0.05 | 0.07 → 0.07 | 0.14 → 0.14 | 0.10 → 0.10 |
| multi_step | 0.00 → 0.05 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 |
| no_call_needed | 0.17 → 0.17 | 0.17 → 0.17 | 0.38 → 0.38 | 0.30 → 0.30 |
| single_call | 0.36 → 0.79 | 0.44 → 0.74 | 0.28 → 0.84 | 0.36 → 0.84 |

## Paired tests over per-task means (sign-flip permutation; * p < 0.05)

| Comparison | success strict | success lenient | F1 strict | F1 lenient |
|---|---|---|---|---|
| run_A → run_B | -0.037 (p 0.143) | -0.050 (p 0.049) * | -0.013 (p 0.665) | -0.021 (p 0.466) |
| base → run_B | +0.041 (p 0.261) | +0.043 (p 0.235) | +0.060 (p 0.062) | +0.067 (p 0.033) * |
| distill → run_B | +0.031 (p 0.075) | +0.057 (p 0.001) * | +0.029 (p 0.079) | +0.042 (p 0.002) * |
| base → run_A | +0.078 (p 0.079) | +0.093 (p 0.029) * | +0.073 (p 0.103) | +0.088 (p 0.040) * |
| base → distill | +0.009 (p 0.795) | -0.015 (p 0.673) | +0.031 (p 0.240) | +0.025 (p 0.354) |

Rollouts that fail strictly but succeed leniently: base 37, distill 24, run_A 45, run_B 38
Strict re-match mismatches vs reported F1: {'base': 0, 'distill': 0, 'run_A': 0, 'run_B': 0}
