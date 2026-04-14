# DOS empirical validation — FIXED carry↔possession link

_Recomputed from `dos_validation_raw.csv` after fixing the DFL carry/possession linkage via frame-range containment. No DOS values were recomputed; only `parent_possession_xg` was corrected for carry rows (the XML omits carries from `TeamPossession > PossessionEvent`, so the naive id join returned 0.0 for every carry)._

- Events evaluated: **6,732** (passes=5128, carries=1604)
- Events with `parent_possession_xg > 0` after fix: **979** (was 718 before; carries now properly linked).

## Mann-Whitney U (fixed)

| Outcome | n+ | n− | mean (+) | mean (−) | median (+) | median (−) | rbc | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| led_to_chance (xg>0) — ALL events | 979 | 5753 | 0.0131 | 0.0116 | 0.0111 | 0.0091 | +0.082 | 2.16e-05 |
| back_line_break — ALL events | 70 | 6662 | 0.0041 | 0.0119 | 0.0006 | 0.0094 | -0.464 | 1.00e+00 |
| led_to_chance — passes only | 718 | 6014 | 0.0132 | 0.0116 | 0.0112 | 0.0092 | +0.061 | 3.94e-03 |
| passes only — led_to_chance | 718 | 4410 | 0.0132 | 0.0113 | 0.0112 | 0.0083 | +nan | 1.14e-04 |
| carries only — led_to_chance | 261 | 1343 | 0.0129 | 0.0124 | 0.0111 | 0.0105 | +nan | 9.70e-02 |

## Correlation DOS ↔ parent xG (continuous)

- Pearson  r = **+0.0274**, p = 2.44e-02
- Spearman ρ = **+0.0514**, p = 2.51e-05
- N = 6728

## Quintiles (fixed)

```
          n  dos_mean  success_rate  line_break_rate  chance_rate   xg_mean
dos_q                                                                      
Q1     1347 -0.001313      0.747587         0.031923     0.118040  0.011935
Q2     1346  0.003653      0.667162         0.006686     0.134473  0.015046
Q3     1346  0.009367      0.612927         0.005944     0.141902  0.011735
Q4     1346  0.016271      0.623328         0.004458     0.167162  0.017615
Q5     1347  0.030926      0.678545         0.002970     0.165553  0.018888
```

![quintiles](dos_validation_quintiles_fixed.png)

## Direction class (fixed)

```
                    n  dos_mean  success_rate  line_break_rate  chance_rate  awareness_mean
direction_class                                                                            
forward          1098  0.012100      0.417122         0.032787     0.137523        0.400057
diagonal         2203  0.012112      0.618702         0.014980     0.158420        0.413783
sideways         3431  0.011468      0.775867         0.000291     0.139609        0.460959
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
