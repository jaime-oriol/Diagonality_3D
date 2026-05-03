# Diagonality 3D — AWS pipeline summary

Single-page index of every artefact produced by `scripts/aws_pipeline.py`. All numbers reproducible from the cached event-level CSV `dos_validation_full.csv`.


## Headline cifras (storytelling)

- **Diagonal pos-DOS rate**: 90.4% vs forward 92.0% vs sideways 92.5%.
- **Mean xT-delta by direction**: diagonal 0.0015, forward 0.0019, sideways -0.0017.
- **Bi-axial disruption (D-Def PC1+PC2 balance)**: diagonal 0.4345, forward 0.4441 — diagonals break BOTH axes simultaneously.
- **Defender awareness (cognitive layer)**: diagonals find defenders 0.4163 vs forwards 0.4042 (lower awareness = more blindness exploited).
- **Nearest defender in his blind half**: diagonals 25.7% vs forward 28.5%.
- **Wrongfooted defenders per action** (>60° misaligned): diagonals 1.6391 vs forwards 1.9613.
- **Half-space vs wing origin** (SV thesis): DOS 0.0119 from half-space vs 0.0122 from wing; xT-delta 0.0001 vs -0.0000.
- **Diagonal from half-space vs from wing**: DOS 0.0121 vs 0.0126 (n=953 vs 814).
- **Top team by mean DOS**: VfB Stuttgart (mean DOS 0.0129, n=705 events, diagonal share 35.6%).
- **Top 3 players by mean DOS** (min 30 events): J. Leweling (VfB Stuttgart, 0.0169), Tiago Tomás (VfB Stuttgart, 0.0162), Luis Díaz (FC Bayern München, 0.0159).

## Direction-class breakdown

| direction_class | n | dos_mean | dos_pos_rate | awareness_mean | xt_delta_mean | ddef_3s_mean | biaxial_mean | n_wrongfooted_mean | pct_nearest_blind | success_rate | back_line_break_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diagonal | 2272 | 0.0121 | 0.9040 | 0.4163 | 0.0015 | 4.4411 | 0.4345 | 1.6391 | 0.2569 | 0.6303 | 0.0145 |
| forward | 1137 | 0.0121 | 0.9200 | 0.4042 | 0.0019 | 4.6105 | 0.4441 | 1.9613 | 0.2851 | 0.4371 | 0.0317 |
| sideways | 3514 | 0.0114 | 0.9249 | 0.4633 | -0.0017 | 4.3198 | 0.4340 | 1.0666 | 0.2477 | 0.7812 | 0.0003 |

## Zone breakdown (Spielverlagerung half-spaces)

Origin zone — where the action begins:
| zone | n | dos_mean | awareness_mean | xt_delta_mean | diag_share | success_rate | back_line_break_rate |
|---|---|---|---|---|---|---|---|
| center | 1248 | 0.0105 | 0.4296 | -0.0006 | 0.4046 | 0.6819 | 0.0072 |
| half_space | 2654 | 0.0119 | 0.4396 | 0.0001 | 0.3591 | 0.6707 | 0.0109 |
| wing | 3021 | 0.0122 | 0.4405 | -0.0000 | 0.2694 | 0.6763 | 0.0106 |

Destination zone:
| zone | n | dos_mean | awareness_mean | xt_delta_mean | diag_share | success_rate | back_line_break_rate |
|---|---|---|---|---|---|---|---|
| center | 1164 | 0.0125 | 0.4474 | 0.0001 | 0.3058 | 0.6607 | 0.0043 |
| half_space | 2713 | 0.0131 | 0.4430 | -0.0001 | 0.2967 | 0.6646 | 0.0077 |
| wing | 3046 | 0.0102 | 0.4304 | -0.0002 | 0.3647 | 0.6901 | 0.0144 |

Zone × direction interaction (the SV signature combination):
| origin_zone | direction_class | n | dos_mean | xt_delta_mean | awareness_mean | success_rate |
|---|---|---|---|---|---|---|
| center | diagonal | 505 | 0.0112 | 0.0004 | 0.3974 | 0.6139 |
| center | forward | 201 | 0.0139 | 0.0019 | 0.3851 | 0.4478 |
| center | sideways | 542 | 0.0085 | -0.0025 | 0.4763 | 0.8321 |
| half_space | diagonal | 953 | 0.0121 | 0.0015 | 0.4158 | 0.6443 |
| half_space | forward | 453 | 0.0132 | 0.0021 | 0.3999 | 0.4459 |
| half_space | sideways | 1248 | 0.0113 | -0.0017 | 0.4721 | 0.7724 |
| wing | diagonal | 814 | 0.0126 | 0.0021 | 0.4286 | 0.6241 |
| wing | forward | 483 | 0.0103 | 0.0017 | 0.4162 | 0.4244 |
| wing | sideways | 1724 | 0.0125 | -0.0015 | 0.4529 | 0.7715 |

## Top players

### By mean DOS (min 30 events)
| player_name | team_name | player_position | n_events | dos_mean | diag_share | xt_delta_mean | ddef_3s_mean |
|---|---|---|---|---|---|---|---|
| J. Leweling | VfB Stuttgart | DLM | 55 | 0.0169 | 0.2364 | 0.0036 | 4.2330 |
| Tiago Tomás | VfB Stuttgart | OHR | 45 | 0.0162 | 0.3333 | 0.0016 | 4.5121 |
| Luis Díaz | FC Bayern München | OLM | 158 | 0.0159 | 0.2215 | -0.0001 | 4.2505 |
| A. Schäfer | 1. FC Union Berlin | HR | 33 | 0.0158 | 0.2727 | -0.0028 | 4.6562 |
| N. Jackson | FC Bayern München | STZ | 40 | 0.0152 | 0.2000 | -0.0049 | 4.3985 |
| Carney Chukwuemeka | Borussia Dortmund | OHL | 53 | 0.0152 | 0.3208 | 0.0002 | 4.8820 |
| M. Beier | Borussia Dortmund | DLM | 49 | 0.0152 | 0.3061 | 0.0001 | 4.6136 |
| A. Stiller | VfB Stuttgart | DML | 113 | 0.0147 | 0.3363 | -0.0002 | 4.2022 |
| Can Uzun | Eintracht Frankfurt | HL | 109 | 0.0146 | 0.2844 | -0.0008 | 4.2460 |
| Serge Gnabry | FC Bayern München | ZO | 117 | 0.0146 | 0.2308 | 0.0012 | 3.7566 |
| Leon Goretzka | FC Bayern München | DML | 120 | 0.0145 | 0.3417 | -0.0006 | 4.2996 |
| Chema Andrés | VfB Stuttgart | DMR | 46 | 0.0145 | 0.3261 | -0.0009 | 4.4676 |
| L. Karl | FC Bayern München | ZO | 78 | 0.0144 | 0.1795 | -0.0001 | 3.9998 |
| R. Khedira | 1. FC Union Berlin | HL | 39 | 0.0143 | 0.3846 | -0.0050 | 4.2856 |
| A. Pavlović | FC Bayern München | DML | 211 | 0.0142 | 0.2796 | 0.0004 | 3.7917 |

### By total xT-delta gained
| player_name | team_name | n_events | xt_delta_sum | xt_delta_mean | dos_mean | diag_share |
|---|---|---|---|---|---|---|
| Joshua Kimmich | FC Bayern München | 457 | 0.4993 | 0.0011 | 0.0129 | 0.3567 |
| T. Bischof | FC Bayern München | 66 | 0.2304 | 0.0035 | 0.0118 | 0.3182 |
| J. Leweling | VfB Stuttgart | 55 | 0.1988 | 0.0036 | 0.0169 | 0.2364 |
| Oliver Burke | 1. FC Union Berlin | 32 | 0.1852 | 0.0058 | 0.0140 | 0.2188 |
| N. Brown | Eintracht Frankfurt | 142 | 0.1832 | 0.0013 | 0.0107 | 0.2394 |
| Dayot Upamecano | FC Bayern München | 312 | 0.1706 | 0.0005 | 0.0113 | 0.4263 |
| Serge Gnabry | FC Bayern München | 117 | 0.1443 | 0.0012 | 0.0146 | 0.2308 |
| Serhou Guirassy | Borussia Dortmund | 34 | 0.1416 | 0.0042 | 0.0139 | 0.2647 |
| Jonathan Tah | FC Bayern München | 370 | 0.1241 | 0.0003 | 0.0113 | 0.4054 |
| Arthur Theate | Eintracht Frankfurt | 241 | 0.1215 | 0.0005 | 0.0101 | 0.3734 |

### By mean D-Def 3s disruption caused
| player_name | team_name | n_events | ddef_3s_mean | biaxial_mean | dos_mean |
|---|---|---|---|---|---|
| Oliver Burke | 1. FC Union Berlin | 32 | 5.9801 | 0.4493 | 0.0140 |
| A. Kemlein | 1. FC Union Berlin | 34 | 5.7886 | 0.4023 | 0.0074 |
| W. Anton | Borussia Dortmund | 53 | 5.5446 | 0.4654 | 0.0077 |
| Serhou Guirassy | Borussia Dortmund | 34 | 5.5009 | 0.4257 | 0.0139 |
| J. Brandt | Borussia Dortmund | 34 | 5.3478 | 0.3680 | 0.0122 |
| L. Querfeld | 1. FC Union Berlin | 71 | 5.3325 | 0.5428 | 0.0080 |
| E. Can | Borussia Dortmund | 69 | 5.2523 | 0.4937 | 0.0071 |
| J. Haberer | 1. FC Union Berlin | 52 | 5.0804 | 0.4554 | 0.0116 |
| W. Mikelbrencis | Hamburger SV | 35 | 5.0460 | 0.4034 | 0.0117 |
| Ellyes Skhiri | Eintracht Frankfurt | 57 | 5.0142 | 0.5211 | 0.0077 |

### By mean wrongfooted defenders (theta)
| player_name | team_name | n_events | n_wrongfooted_mean | pct_nearest_blind | dos_mean |
|---|---|---|---|---|---|
| Frederik Rønnow | 1. FC Union Berlin | 98 | 3.1122 | 0.5816 | 0.0121 |
| A. Nübel | VfB Stuttgart | 51 | 2.2941 | 0.5294 | 0.0091 |
| R. Khedira | 1. FC Union Berlin | 39 | 2.1538 | 0.3514 | 0.0143 |
| Fábio Vieira | Hamburger SV | 30 | 2.0667 | 0.4483 | 0.0131 |
| Hugo Larsson | Eintracht Frankfurt | 79 | 2.0253 | 0.2436 | 0.0141 |
| Danilho Doekhi | 1. FC Union Berlin | 45 | 2.0222 | 0.4146 | 0.0089 |
| A. Kemlein | 1. FC Union Berlin | 34 | 1.9706 | 0.4062 | 0.0074 |
| Can Uzun | Eintracht Frankfurt | 109 | 1.9541 | 0.3010 | 0.0146 |
| L. Assignon | VfB Stuttgart | 57 | 1.9123 | 0.2909 | 0.0137 |
| J. Haberer | 1. FC Union Berlin | 52 | 1.8654 | 0.4038 | 0.0116 |

### By LOWEST defender awareness exploited (cognitive layer)
| player_name | team_name | n_events | awareness_mean | detection_delay_mean_s | dos_mean |
|---|---|---|---|---|---|
| Frederik Rønnow | 1. FC Union Berlin | 98 | 0.2855 | 0.2062 | 0.0121 |
| A. Nübel | VfB Stuttgart | 51 | 0.3220 | 0.2242 | 0.0091 |
| Manuel Neuer | FC Bayern München | 151 | 0.3692 | 0.1836 | 0.0070 |
| D. Heuer Fernandes | Hamburger SV | 52 | 0.3729 | 0.1823 | 0.0075 |
| Kauã Santos | Eintracht Frankfurt | 91 | 0.3800 | 0.1815 | 0.0062 |
| L. Assignon | VfB Stuttgart | 57 | 0.3881 | 0.1890 | 0.0137 |
| A. Kemlein | 1. FC Union Berlin | 34 | 0.3928 | 0.1629 | 0.0074 |
| L. Querfeld | 1. FC Union Berlin | 71 | 0.4006 | 0.1511 | 0.0080 |
| Tom Rothe | 1. FC Union Berlin | 42 | 0.4016 | 0.1578 | 0.0104 |
| Danilho Doekhi | 1. FC Union Berlin | 45 | 0.4037 | 0.1739 | 0.0089 |

### By diagonal share of all actions
| player_name | team_name | n_events | diag_share | fwd_share | dos_mean |
|---|---|---|---|---|---|
| Manuel Neuer | FC Bayern München | 151 | 0.5099 | 0.2185 | 0.0070 |
| A. Nübel | VfB Stuttgart | 51 | 0.5098 | 0.3529 | 0.0091 |
| W. Omari | Hamburger SV | 55 | 0.4909 | 0.1273 | 0.0119 |
| Nico Schlotterbeck | Borussia Dortmund | 62 | 0.4677 | 0.2419 | 0.0087 |
| N. Remberg | Hamburger SV | 44 | 0.4318 | 0.0227 | 0.0085 |
| F. Jeltsch | VfB Stuttgart | 77 | 0.4286 | 0.1429 | 0.0077 |
| Dayot Upamecano | FC Bayern München | 312 | 0.4263 | 0.1410 | 0.0113 |
| Jeff Chabot | VfB Stuttgart | 65 | 0.4154 | 0.1692 | 0.0100 |
| J. Brandt | Borussia Dortmund | 34 | 0.4118 | 0.1471 | 0.0122 |
| Jonathan Tah | FC Bayern München | 370 | 0.4054 | 0.1784 | 0.0113 |

## Teams summary

| team_name | n_events | dos_mean | xt_delta_mean | ddef_3s_mean | biaxial_mean | diag_share | success_rate |
|---|---|---|---|---|---|---|---|
| VfB Stuttgart | 705 | 0.0129 | 0.0005 | 4.4422 | 0.4402 | 0.3560 | 0.6667 |
| FC Bayern München | 2961 | 0.0126 | -0.0001 | 4.2021 | 0.4370 | 0.3279 | 0.7133 |
| 1. FC Union Berlin | 658 | 0.0112 | -0.0011 | 4.8251 | 0.4480 | 0.3222 | 0.5699 |
| Borussia Dortmund | 577 | 0.0107 | 0.0002 | 4.8580 | 0.4123 | 0.3328 | 0.6395 |
| Eintracht Frankfurt | 1536 | 0.0106 | -0.0000 | 4.4209 | 0.4389 | 0.3138 | 0.6771 |
| Hamburger SV | 486 | 0.0101 | 0.0000 | 4.4635 | 0.4243 | 0.3374 | 0.6337 |

## Match summary

| match | n_events | dos_mean | diag_share | top_dos_player | top_dos_value |
|---|---|---|---|---|---|
| Bayern_Hamburg | 1480 | 0.0122 | 0.3311 | L. Karl | 0.0188 |
| Dortmund_Stuttgart | 1282 | 0.0119 | 0.3456 | Karim Adeyemi | 0.0199 |
| Frankfurt_Bayern | 1527 | 0.0108 | 0.3039 | Luis Díaz | 0.0199 |
| Frankfurt_Union | 1249 | 0.0118 | 0.3355 | Jean-Mattéo Bahoya | 0.0167 |
| Union_Bayern | 1385 | 0.0121 | 0.3292 | Serge Gnabry | 0.0191 |

## Top events selected for video

Composite score = `DOS + 5·xT_delta + 0.05·back_line_break` — picks events that combine mechanistic interest (DOS) with real impact (xT-delta and direct back-line breaks). Pure DOS-only ranking is in `outputs/tables/top_events_full_vector.csv`.

_(see `outputs/tables/top_dos_events.json` for full metadata, 10 entries)_

| # | match | player | type | frame | DOS | xT-delta | back-line break |
|---:|---|---|---|---:|---:|---:|---|
| 1 | Frankfurt_Bayern | F. Chaïbi | pass | 3595150 | -0.0000 | 0.2227 | no |
| 2 | Frankfurt_Bayern | Serge Gnabry | pass | 3332702 | 0.0163 | 0.1832 | no |
| 3 | Frankfurt_Union | A. Ilić | pass | 3011270 | 0.0157 | 0.1786 | no |
| 4 | Union_Bayern | T. Bischof | pass | 3131617 | 0.0187 | 0.1770 | no |
| 5 | Bayern_Hamburg | Luis Díaz | carry | 3393558 | 0.0015 | 0.1771 | no |
| 6 | Frankfurt_Union | N. Brown | pass | 2917679 | 0.0343 | 0.1617 | no |
| 7 | Bayern_Hamburg | Joshua Kimmich | pass | 3613622 | 0.0059 | 0.1656 | no |
| 8 | Frankfurt_Union | A. Knauff | pass | 3087788 | 0.0181 | 0.1238 | no |
| 9 | Bayern_Hamburg | L. Karl | pass | 3639170 | 0.0208 | 0.1227 | no |
| 10 | Bayern_Hamburg | Harry Kane | pass | 3395284 | 0.0073 | 0.1128 | YES |

## Files index (every artefact)


### Reports

### Tables (CSV)
- [direction_breakdown.csv](tables/direction_breakdown.csv)
- [events_per_player_match.csv](tables/events_per_player_match.csv)
- [matches_summary.csv](tables/matches_summary.csv)
- [matches_team_dir_breakdown.csv](tables/matches_team_dir_breakdown.csv)
- [players_by_awareness_low.csv](tables/players_by_awareness_low.csv)
- [players_by_ddef.csv](tables/players_by_ddef.csv)
- [players_by_diag_share.csv](tables/players_by_diag_share.csv)
- [players_by_dos.csv](tables/players_by_dos.csv)
- [players_by_volume.csv](tables/players_by_volume.csv)
- [players_by_wrongfooting.csv](tables/players_by_wrongfooting.csv)
- [players_by_xt.csv](tables/players_by_xt.csv)
- [teams_summary.csv](tables/teams_summary.csv)
- [top_events_full_vector.csv](tables/top_events_full_vector.csv)
- [zones_dest_breakdown.csv](tables/zones_dest_breakdown.csv)
- [zones_origin_breakdown.csv](tables/zones_origin_breakdown.csv)
- [zones_x_direction.csv](tables/zones_x_direction.csv)

### Videos (26)
- [top01_Frankfurt_Bayern_F._Chaïbi_pass_3595150.mp4](videos/top01_Frankfurt_Bayern_F._Chaïbi_pass_3595150.mp4) (7.2 MB)
- [top01_ppcf_Frankfurt_Bayern_F._Chaïbi_pass_3595150.mp4](videos/top01_ppcf_Frankfurt_Bayern_F._Chaïbi_pass_3595150.mp4) (13.1 MB)
- [top01_vision_Frankfurt_Bayern_F._Chaïbi_pass_3595150.mp4](videos/top01_vision_Frankfurt_Bayern_F._Chaïbi_pass_3595150.mp4) (6.4 MB)
- [top02_Frankfurt_Bayern_Serge_Gnabry_pass_3332702.mp4](videos/top02_Frankfurt_Bayern_Serge_Gnabry_pass_3332702.mp4) (5.6 MB)
- [top02_ppcf_Frankfurt_Bayern_Serge_Gnabry_pass_3332702.mp4](videos/top02_ppcf_Frankfurt_Bayern_Serge_Gnabry_pass_3332702.mp4) (8.6 MB)
- [top02_vision_Frankfurt_Bayern_Serge_Gnabry_pass_3332702.mp4](videos/top02_vision_Frankfurt_Bayern_Serge_Gnabry_pass_3332702.mp4) (8.9 MB)
- [top03_Frankfurt_Union_A._Ilić_pass_3011270.mp4](videos/top03_Frankfurt_Union_A._Ilić_pass_3011270.mp4) (5.4 MB)
- [top03_ppcf_Frankfurt_Union_A._Ilić_pass_3011270.mp4](videos/top03_ppcf_Frankfurt_Union_A._Ilić_pass_3011270.mp4) (8.9 MB)
- [top03_vision_Frankfurt_Union_A._Ilić_pass_3011270.mp4](videos/top03_vision_Frankfurt_Union_A._Ilić_pass_3011270.mp4) (10.2 MB)
- [top04_Union_Bayern_T._Bischof_pass_3131617.mp4](videos/top04_Union_Bayern_T._Bischof_pass_3131617.mp4) (5.6 MB)
- [top04_ppcf_Union_Bayern_T._Bischof_pass_3131617.mp4](videos/top04_ppcf_Union_Bayern_T._Bischof_pass_3131617.mp4) (8.1 MB)
- [top04_vision_Union_Bayern_T._Bischof_pass_3131617.mp4](videos/top04_vision_Union_Bayern_T._Bischof_pass_3131617.mp4) (8.8 MB)
- [top05_Bayern_Hamburg_Luis_Díaz_carry_3393558.mp4](videos/top05_Bayern_Hamburg_Luis_Díaz_carry_3393558.mp4) (7.6 MB)
- [top05_ppcf_Bayern_Hamburg_Luis_Díaz_carry_3393558.mp4](videos/top05_ppcf_Bayern_Hamburg_Luis_Díaz_carry_3393558.mp4) (13.0 MB)
- [top05_vision_Bayern_Hamburg_Luis_Díaz_carry_3393558.mp4](videos/top05_vision_Bayern_Hamburg_Luis_Díaz_carry_3393558.mp4) (10.8 MB)
- [top06_Frankfurt_Union_N._Brown_pass_2917679.mp4](videos/top06_Frankfurt_Union_N._Brown_pass_2917679.mp4) (5.4 MB)
- [top06_ppcf_Frankfurt_Union_N._Brown_pass_2917679.mp4](videos/top06_ppcf_Frankfurt_Union_N._Brown_pass_2917679.mp4) (8.1 MB)
- [top06_vision_Frankfurt_Union_N._Brown_pass_2917679.mp4](videos/top06_vision_Frankfurt_Union_N._Brown_pass_2917679.mp4) (9.2 MB)
- [top07_Bayern_Hamburg_Joshua_Kimmich_pass_3613622.mp4](videos/top07_Bayern_Hamburg_Joshua_Kimmich_pass_3613622.mp4) (6.3 MB)
- [top07_ppcf_Bayern_Hamburg_Joshua_Kimmich_pass_3613622.mp4](videos/top07_ppcf_Bayern_Hamburg_Joshua_Kimmich_pass_3613622.mp4) (8.8 MB)
- [top07_vision_Bayern_Hamburg_Joshua_Kimmich_pass_3613622.mp4](videos/top07_vision_Bayern_Hamburg_Joshua_Kimmich_pass_3613622.mp4) (9.1 MB)
- [top08_Frankfurt_Union_A._Knauff_pass_3087788.mp4](videos/top08_Frankfurt_Union_A._Knauff_pass_3087788.mp4) (6.3 MB)
- [top08_ppcf_Frankfurt_Union_A._Knauff_pass_3087788.mp4](videos/top08_ppcf_Frankfurt_Union_A._Knauff_pass_3087788.mp4) (10.1 MB)
- [top08_vision_Frankfurt_Union_A._Knauff_pass_3087788.mp4](videos/top08_vision_Frankfurt_Union_A._Knauff_pass_3087788.mp4) (12.0 MB)
- [top09_Bayern_Hamburg_L._Karl_pass_3639170.mp4](videos/top09_Bayern_Hamburg_L._Karl_pass_3639170.mp4) (6.4 MB)
- [top10_Bayern_Hamburg_Harry_Kane_pass_3395284.mp4](videos/top10_Bayern_Hamburg_Harry_Kane_pass_3395284.mp4) (5.6 MB)

### Frame galleries
- DOS frames: 25 files in `outputs/frames/`
- PPCF frames: 25 files in `outputs/frames/`
- Pass maps: 10 files in `outputs/frames/pass_maps/`

#### Pass maps rendered
- [A._Schäfer_Union_Bayern.png](frames/pass_maps/A._Schäfer_Union_Bayern.png)
- [A._Stiller_Dortmund_Stuttgart.png](frames/pass_maps/A._Stiller_Dortmund_Stuttgart.png)
- [Carney_Chukwuemeka_Dortmund_Stuttgart.png](frames/pass_maps/Carney_Chukwuemeka_Dortmund_Stuttgart.png)
- [J._Leweling_Dortmund_Stuttgart.png](frames/pass_maps/J._Leweling_Dortmund_Stuttgart.png)
- [Luis_Díaz_Bayern_Hamburg.png](frames/pass_maps/Luis_Díaz_Bayern_Hamburg.png)
- [Luis_Díaz_Frankfurt_Bayern.png](frames/pass_maps/Luis_Díaz_Frankfurt_Bayern.png)
- [Luis_Díaz_Union_Bayern.png](frames/pass_maps/Luis_Díaz_Union_Bayern.png)
- [M._Beier_Dortmund_Stuttgart.png](frames/pass_maps/M._Beier_Dortmund_Stuttgart.png)
- [N._Jackson_Bayern_Hamburg.png](frames/pass_maps/N._Jackson_Bayern_Hamburg.png)
- [Tiago_Tomás_Dortmund_Stuttgart.png](frames/pass_maps/Tiago_Tomás_Dortmund_Stuttgart.png)