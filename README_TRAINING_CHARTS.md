# Training Charts Report

This report contains PNG charts generated from the milestone table in [README_TRAINING_ANALYSIS.md](README_TRAINING_ANALYSIS.md).

## Data Source
- Source: milestone rows from your training-history analysis.
- Coverage: major turning points from episode 50 to episode 17000.
- Purpose: communicate regime transitions clearly (dominance, collapse, recovery).

## Charts

### 1) Robber Win Rate (Milestones)

![Robber win-rate milestones](reports/assets/training_winrate_milestones.png)

### 2) ELO Diff (Milestones)

![ELO diff milestones](reports/assets/training_elo_milestones.png)

## How To Regenerate

```bash
conda activate cv_conda
python tools/generate_training_charts.py
```

## Notes
- These charts are milestone-based, not per-episode dense curves.
- They are designed for README communication and checkpoint strategy review.
