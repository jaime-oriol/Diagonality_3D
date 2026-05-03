#!/usr/bin/env bash
# Incremental sync from EC2 to aws_results/. Run as many times as needed —
# rsync --update only pulls newer files. Pulls EVERYTHING useful: caches
# (per-match preprocess), validation CSVs, reports, tables, frames, videos,
# kane goal renders, olise pass map, master log + state.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
KEY="$HOME/aws/tfg-key.pem"
HOST="ec2-user@18.199.146.84"
SSH_OPT="-i $KEY -o StrictHostKeyChecking=accept-new"

t0=$(date +%s)

echo "=== sync outputs/ tree (tables, frames, videos, reports, SUMMARY.md, log) ==="
rsync -a --partial --update --info=stats1 \
    -e "ssh $SSH_OPT" \
    "$HOST:~/Diagonality_3D/outputs/" "$HERE/" 2>&1 | tail -3

echo ""
echo "=== sync intermediate validation CSVs (test/dos_validation_*.csv) ==="
rsync -a --partial --update --info=stats1 \
    --include='dos_validation_*.csv' --exclude='*' \
    -e "ssh $SSH_OPT" \
    "$HOST:~/Diagonality_3D/test/" "$HERE/intermediate/csvs/" 2>&1 | tail -3

echo ""
echo "=== sync intermediate reports (test/dos_validation_report*.md + axes/quintile PNGs) ==="
rsync -a --partial --update --info=stats1 \
    --include='dos_validation_report*.md' --include='dos_validation_*.png' --exclude='*' \
    -e "ssh $SSH_OPT" \
    "$HOST:~/Diagonality_3D/test/" "$HERE/intermediate/reports/" 2>&1 | tail -3

echo ""
echo "=== sync kane goal renders + olise pass map (intermediate test/*.mp4 + .png) ==="
rsync -a --partial --update --info=stats1 \
    --include='*kane*.mp4' --include='*kane*.png' --include='passes_olise_*.png' --exclude='*' \
    -e "ssh $SSH_OPT" \
    "$HOST:~/Diagonality_3D/test/" "$HERE/intermediate/kane_olise/" 2>&1 | tail -3

echo ""
echo "=== sync FULL CACHE (5 matches, ~5 GB total — only changed files) ==="
rsync -a --partial --update --info=stats1 \
    -e "ssh $SSH_OPT" \
    "$HOST:~/Diagonality_3D/cache/" "$HERE/cache/" 2>&1 | tail -3

echo ""
echo "=== final structure ==="
find "$HERE" -maxdepth 2 -type d | sort
echo ""
echo "=== file count + size per folder ==="
for d in "$HERE"/*/; do
    n=$(find "$d" -type f | wc -l)
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    printf "  %-30s %4d files  %6s\n" "${d##*/aws_results/}" "$n" "$sz"
done
echo "  ----- root file:"
ls -la "$HERE"/*.json "$HERE"/*.md "$HERE"/*.txt 2>/dev/null | awk '{printf "  %-30s          %6s\n", $NF, $5}'
echo "  Total aws_results/: $(du -sh "$HERE" | cut -f1)"
dt=$(($(date +%s) - t0))
echo "  Sync took: ${dt}s"
