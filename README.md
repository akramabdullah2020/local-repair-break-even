# When Does Local Path Repair Pay Off?

Code and data for the paper **"When Does Local Path Repair Pay Off? A Break-Even Analysis of Local Repair versus Global and Incremental Replanning for Mobile Robots with a Limited Perception Range"**.

When a robot perceives a new obstacle on its path, it can repair the invalidated segment locally or replan to the goal (from scratch with A\*, or incrementally with D\* Lite). This repository contains:

- a **break-even model**: attempting repair first pays off when `p_fail < 1 - C_repair / C_fallback`;
- a **paired event-level evaluation**: every strategy is measured on the identical robot state at every blocking event, with D\* Lite's deferred bookkeeping charged in full;
- the experiment: 24,960 episodes on 9 map families, including the Moving AI benchmark maps;
- the analysis, figures and raw event tables.

## Key results (held-out split: 17,472 episodes, 48,422 blocking events)

| Policy | Work relative to D\* Lite (95% CI) |
|---|---|
| Unbounded repair → D\* Lite | **0.651** [0.639, 0.663] |
| Oracle, perception-bounded repair → D\* Lite | 0.877 [0.870, 0.882] |
| Learned failure predictor → D\* Lite | 0.935 [0.931, 0.939] |
| D\* Lite at every event | 1.000 |
| Perception-bounded repair → D\* Lite | 1.066 [1.054, 1.078] |
| A\* at every event | 5.237 [5.074, 5.407] |

- The break-even condition identifies the cheaper strategy in **97.6%** (vs. A\*) and **90.8%** (vs. D\* Lite) of 576 experimental cells.
- Failure predictor AUC **0.95**; a hand-crafted composite risk score: **0.46** (uninformative).
- Work = edge-cost evaluations (hardware independent). See `results/v2_analysis/summary.json`.

## Repository layout

```
src/
  environment.py         grid map and square perception window
  planners/              A*, D* Lite (optimized), local splice repair + risk score
  validation.py          independent motion validator (adjacency, occupancy, corner cutting)
  scenarios_v2.py        scenario generator: 9 map families x 4 blockage types
  movingai.py            Moving AI .map reader and cropping
  event_eval.py          paired event-level evaluation (D* Lite shadow instances)
  policies.py            decision policies, failure predictor, break-even tables
run_events.py            main experiment (parallel)
fit_predictor.py         fit the failure predictor on the training split
analyze_v2.py            statistics and figures
run.py                   closed-loop benchmark on the original 5 scenarios
scripts/download_movingai.py
tests/                   unit, differential and conservation tests
results/v2_events/       raw event tables (pilot, training split, held-out split)
results/v2_analysis/     tables, figures, summary.json
paper/                   LaTeX source of the paper
```

## Reproducing the results

Requires Python 3.11+.

```bash
pip install -r requirements.txt
python scripts/download_movingai.py            # Moving AI maps -> data/movingai/ (~70 MB)
python -m pytest -q tests                      # 37 tests, ~10 s

# Stage 1: training split (9 seeds, 7,488 episodes; ~35 min on 7 cores)
python run_events.py --split train --out results/v2_events/full_train
python fit_predictor.py results/v2_events/full_train/events.csv results/v2_events/predictor.pkl

# Stage 2: held-out split (21 seeds, 17,472 episodes; ~90 min on 7 cores)
python run_events.py --split eval --predictor results/v2_events/predictor.pkl --out results/v2_events/full_eval

# Statistics and figures
python analyze_v2.py --eval results/v2_events/full_eval --train results/v2_events/full_train
```

Seeds with `seed % 10 < 3` form the training split. Everything is deterministic given the seed except wall-clock timings. Each `metadata.json` records the SHA-256 of every source file used.

To run the larger D\* Lite differential test: `DSTAR_DIFF_CASES=1000 python -m pytest tests/test_week2_dstar_differential.py`.

## Data

Moving AI maps are **not redistributed** here; `scripts/download_movingai.py` fetches them from <https://movingai.com/benchmarks/grids.html>. Please cite N. R. Sturtevant, "Benchmarks for Grid-Based Pathfinding," *IEEE TCIAIG*, 4(2):144–148, 2012.

## Citation

See `CITATION.cff`. A DOI will be added on archival at Zenodo.

## License

MIT, see `LICENSE`.
