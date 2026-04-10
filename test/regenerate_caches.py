"""Regenerate the per-match caches with the wider PRE_WINDOW.

The 2.5s scanning memory in src/scanning.py needs up to 125 frames of
historical orientations within the same possession segment. preprocess.py
uses PRE_WINDOW_FRAMES = 150 (3s) so the scanning lookback is fully
covered for every event window in the cache.

What this script does (in order, per match):
  1. Run preprocess.preprocess_match(match) which:
       - Loads metadata + events (small)
       - Builds the unified events DataFrame (passes + carries +
         receptions + shots) and writes events.parquet + possessions.parquet
       - Collects all unique frames in [event - 3s, event + 3s] (or for
         carries, [carry_start - 3s, carry_end + 3s])
       - Streams the heavy nested skeleton parquet ROW GROUP BY ROW GROUP,
         keeping only the frames we want, and writes them flat to
         skeleton.parquet. Peak memory ~ 1 row group + the chunk being
         written, ~200 MB max.
       - Extracts ball positions and writes ball.parquet (small).
  2. Explicitly del + gc.collect() between matches so WSL stays cool.
  3. Print a summary of cache sizes per match at the end.

USAGE:
    source ~/anaconda3/bin/activate base
    python3 test/regenerate_caches.py            # all 5 matches
    python3 test/regenerate_caches.py Bayern_Hamburg Frankfurt_Bayern  # subset

Memory model: each match is preprocessed in isolation. The skeleton
streaming inside preprocess._extract_skeleton_batch already releases
each row-group table after writing its chunk, so peak memory stays
flat (not cumulative) within a match. Between matches we drop all
local references and force garbage collection.
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.loader import MATCHES
from src.preprocess import preprocess_match, CACHE_DIR


def _format_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    mb = path.stat().st_size / 1024 / 1024
    return f"{mb:6.1f} MB"


def _print_cache_summary(match: str) -> float:
    """Print sizes of every cache file for `match`. Returns total MB."""
    cache_path = CACHE_DIR / match
    files = ["skeleton.parquet", "ball.parquet",
             "events.parquet", "possessions.parquet", "metadata.json"]
    total = 0.0
    print(f"  cache: {cache_path}")
    for f in files:
        p = cache_path / f
        if p.exists():
            mb = p.stat().st_size / 1024 / 1024
            total += mb
            print(f"    {f:25s} {mb:7.1f} MB")
        else:
            print(f"    {f:25s}  missing")
    print(f"    {'TOTAL':25s} {total:7.1f} MB")
    return total


def main(matches: list) -> None:
    n = len(matches)
    print(f"Regenerating {n} match cache(s) with PRE_WINDOW_FRAMES=150")
    print(f"Cache root: {CACHE_DIR}\n")

    overall_t0 = time.time()
    sizes = {}
    for i, match in enumerate(matches, 1):
        if match not in MATCHES:
            print(f"!! Unknown match '{match}'. Valid: {list(MATCHES)}")
            continue
        print(f"\n{'='*72}")
        print(f"  [{i}/{n}] {match}")
        print(f"{'='*72}")
        t0 = time.time()
        try:
            preprocess_match(match, verbose=True)
        except Exception as exc:  # noqa: BLE001
            print(f"!! ERROR processing {match}: {exc}")
            # Continue with the next match instead of bailing on the batch.
            gc.collect()
            continue
        sizes[match] = _print_cache_summary(match)
        dt = time.time() - t0
        print(f"  ({dt:.0f}s)")
        # Free anything Python may still be holding from this match
        # before moving to the next one. WSL likes this a lot.
        gc.collect()

    overall_dt = time.time() - overall_t0
    print(f"\n{'='*72}")
    print(f"  DONE in {overall_dt/60:.1f} min")
    print(f"{'='*72}")
    if sizes:
        print("\nFinal cache sizes:")
        for match, mb in sizes.items():
            print(f"  {match:25s} {mb:7.1f} MB")
        print(f"  {'TOTAL':25s} {sum(sizes.values()):7.1f} MB")


if __name__ == "__main__":
    args = sys.argv[1:]
    matches = args if args else list(MATCHES.keys())
    main(matches)
