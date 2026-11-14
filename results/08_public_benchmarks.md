# Experiment 08: public benchmarks (When2Call, BFCL-Hi)

| Model | When2Call decision acc. | calls when it should | tool hallucination (cannot answer) | calls when info is missing | BFCL-Hi simple acc. | BFCL-Hi irrelevance acc. |
|---|---|---|---|---|---|---|
| base | 0.345 | 0.985 | 0.975 | 0.975 | 0.264 | 0.097 |
| distill | 0.530 | 0.785 | 0.505 | 0.690 | 0.043 | 0.903 |
| run_A | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | 0.996 |
| run_B | 0.667 | 0.095 | 0.045 | 0.050 | 0.008 | 0.977 |

Paired sign-flip permutation tests on per-question correctness (delta = second − first):

- base_vs_distill: when2call +0.185 (p 0.000 *); bfcl_hi_simple -0.221 (p 0.000 *); bfcl_hi_irrelevance +0.806 (p 0.000 *)
- base_vs_run_A: when2call +0.322 (p 0.000 *); bfcl_hi_simple -0.264 (p 0.000 *); bfcl_hi_irrelevance +0.899 (p 0.000 *)
- base_vs_run_B: when2call +0.322 (p 0.000 *); bfcl_hi_simple -0.256 (p 0.000 *); bfcl_hi_irrelevance +0.880 (p 0.000 *)
- distill_vs_run_B: when2call +0.137 (p 0.000 *); bfcl_hi_simple -0.035 (p 0.023 *); bfcl_hi_irrelevance +0.074 (p 0.000 *)
- run_A_vs_run_B: when2call +0.000 (p 1.000); bfcl_hi_simple +0.008 (p 0.501); bfcl_hi_irrelevance -0.019 (p 0.125)
