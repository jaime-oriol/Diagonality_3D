# DOS empirical validation — FIXED carry↔possession link

_Recomputed from `dos_validation_raw.csv` after fixing the DFL carry/possession linkage via frame-range containment. No DOS values were recomputed; only `parent_possession_xg` was corrected for carry rows (the XML omits carries from `TeamPossession > PossessionEvent`, so the naive id join returned 0.0 for every carry)._

- Events evaluated: **6,923** (passes=5128, carries=1604)
- Events with `parent_possession_xg > 0` after fix: **1,020** (was 718 before; carries now properly linked).

## Mann-Whitney U (fixed)

| Outcome | n+ | n− | mean (+) | mean (−) | median (+) | median (−) | rbc | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| led_to_chance (xg>0) — ALL events | 1020 | 5903 | 0.0130 | 0.0115 | 0.0110 | 0.0091 | +0.079 | 2.67e-05 |
| back_line_break — ALL events | 70 | 6853 | 0.0041 | 0.0118 | 0.0006 | 0.0094 | -0.463 | 1.00e+00 |
| led_to_chance — passes only | 718 | 6205 | 0.0132 | 0.0116 | 0.0112 | 0.0092 | +0.062 | 3.22e-03 |
| passes only — led_to_chance | 718 | 4410 | 0.0132 | 0.0113 | 0.0112 | 0.0083 | +nan | 1.14e-04 |
| carries only — led_to_chance | 261 | 1343 | 0.0129 | 0.0124 | 0.0111 | 0.0105 | +nan | 9.70e-02 |

## Correlation DOS ↔ parent xG (continuous)

- Pearson  r = **+0.0239**, p = 4.73e-02
- Spearman ρ = **+0.0496**, p = 3.83e-05
- N = 6890

## Quintiles (fixed)

```
          n  dos_mean  success_rate  line_break_rate  chance_rate   xg_mean
dos_q                                                                      
Q1     1385 -0.001281      0.753069         0.031769     0.119134  0.011983
Q2     1384  0.003657      0.679191         0.005780     0.135116  0.015106
Q3     1385  0.009332      0.623827         0.005776     0.147292  0.014358
Q4     1384  0.016200      0.634393         0.004335     0.170520  0.018193
Q5     1385  0.030802      0.685199         0.002888     0.164621  0.018589
```

![quintiles](dos_validation_quintiles_fixed.png)

## Direction class (fixed)

```
                    n  dos_mean  success_rate  line_break_rate  chance_rate  awareness_mean
direction_class                                                                            
forward          1137  0.012070      0.437115         0.031662     0.139842        0.404185
diagonal         2272  0.012064      0.630282         0.014525     0.161532        0.416295
sideways         3514  0.011429      0.781161         0.000285     0.140581        0.463324
```

## Logistic regression (passes only, fixed xG for parent)

### Target: `back_line_break`
- n = 4761, Pseudo R² = 0.1566, LLR p = 1.31e-21

| coef | value | p |
|---|---:|---:|
| const | -2.9436 | 2.62e-06 |
| dos | -55.1417 | 6.08e-04 |
| distance | +0.0598 | 1.56e-09 |
| pressure_player | +0.1191 | 7.97e-01 |
| n_def_lane | +0.2291 | 2.80e-02 |
| n_def_goal | -0.2926 | 9.31e-07 |

### Target: `chance`
- n = 4761, Pseudo R² = 0.0116, LLR p = 9.74e-09

| coef | value | p |
|---|---:|---:|
| const | -1.1295 | 8.65e-06 |
| dos | +7.8611 | 2.06e-02 |
| distance | -0.0010 | 8.24e-01 |
| pressure_player | +0.2153 | 1.44e-01 |
| n_def_lane | +0.0384 | 3.27e-01 |
| n_def_goal | -0.0921 | 6.10e-05 |
