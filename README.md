# DIAG — Diagonality

Demostracion computacional de que las acciones diagonales desorganizan la estructura defensiva mas que las ortogonales, y construccion de un pitch control direccional (D-PPCF) que genera mapas de vulnerabilidad diagonal por rival.

TFG Ingenieria Informatica (UFV), curso 2026-2027.

## Pipeline

```
Tracking 25Hz -> estructura defensiva frame-by-frame (Delaunay, convex hull)
    -> clasificacion angular de acciones (relativo a linea defensiva)
    -> D-Def descompuesta PC1 longitudinal / PC2 lateral
    -> DML: curva dosis-respuesta angulo -> disrupcion
    -> D-PPCF: pitch control direccional
    -> DOS: mapas de vulnerabilidad diagonal por rival
```

## Structure

```
DIAG/
├── docs/           # Investigacion, SOTA, articulos de referencia
├── src/            # Source code
├── notebooks/      # Jupyter experiments
└── data/           # Git-ignored (tracking + eventos)
```

## References

- Goes et al. (2019) — D-Def
- Spearman (2018) — PPCF
- Colangelo & Lee (2025) — DML tratamiento continuo
- Spielverlagerung (2025) — Tactical Theory: Diagonality
