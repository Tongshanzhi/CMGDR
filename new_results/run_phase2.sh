#!/usr/bin/env bash
# Phase 2: after multi-seed completes, run sensitivity + stratified ablation
# + analyses.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p new_results/logs

echo "[$(date '+%H:%M:%S')] === PHASE 2: hyper-param sensitivity ==="
python -u new_results/run_sensitivity.py 2>&1 | tee new_results/logs/sensitivity.log

echo "[$(date '+%H:%M:%S')] === PHASE 2: stratified-vs-uniform ablation ==="
python -u new_results/run_stratified_ablation.py 2>&1 | tee new_results/logs/stratified.log

echo "[$(date '+%H:%M:%S')] === PHASE 2: post-hoc analyses ==="
python -u new_results/aggregate_multiseed.py 2>&1 | tee new_results/logs/multiseed_summary.log
python -u new_results/run_slice_analysis.py 2>&1 | tee new_results/logs/slice.log
python -u new_results/run_ipw_dcg.py 2>&1 | tee new_results/logs/ipw.log
python -u new_results/run_efficiency.py 2>&1 | tee new_results/logs/efficiency.log

echo "[$(date '+%H:%M:%S')] === Phase 2 complete ==="
