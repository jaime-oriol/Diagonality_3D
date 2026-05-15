"""Helpers compartidos para leer la cache de skeleton por ventanas de
frames de forma memory-safe.

Extraido de validate_dos_outcomes.py para que la cadena de enrichment
(enrich_with_ddef.py, enrich_with_theta.py) reutilice exactamente la
misma lectura chunked + suavizado de velocidades sin importarse entre si.
"""

import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import savgol_filter

from src.preprocess import CACHE_DIR

# Caps por chunk para acotar el pico de memoria en WSL (8 GB RAM).
DEFAULT_MAX_FRAME_SPAN = 30000   # ~10 min de juego @ 50fps; ~3-5M filas skeleton
DEFAULT_MAX_CHUNK_EVENTS = 250   # tope de seguridad de eventos por chunk

# Savitzky-Golay para vx, vy (mismos valores que usa el render).
SMOOTH_W, POLY = 13, 1
MAX_SPEED = 12.0                 # m/s: por encima de esto la velocidad es ruido


def _read_skeleton_window(match: str, f_lo: int, f_hi: int) -> pd.DataFrame:
    """Lee solo las filas de skeleton en [f_lo, f_hi] usando predicate
    pushdown de parquet — acota el pico de memoria sea cual sea el partido."""
    path = CACHE_DIR / match / "skeleton.parquet"
    table = pq.read_table(
        path,
        filters=[("frame_number", ">=", int(f_lo)),
                 ("frame_number", "<=", int(f_hi))],
    )
    return table.to_pandas()


def _smooth_velocities(dyn: pd.DataFrame) -> pd.DataFrame:
    """Capa outliers de velocidad y aplica Savitzky-Golay por jugador."""
    raw_speed = (dyn["vx"]**2 + dyn["vy"]**2).pow(0.5)
    dyn.loc[raw_speed > MAX_SPEED, ["vx", "vy"]] = float("nan")
    for col in ("vx", "vy"):
        dyn[col] = (
            dyn.groupby(["team", "jersey"], sort=False)[col]
            .transform(lambda s: pd.Series(
                savgol_filter(s.fillna(0).values, SMOOTH_W, POLY),
                index=s.index))
        )
    return dyn
