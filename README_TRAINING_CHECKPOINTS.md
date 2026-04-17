# Training Checkpoint Timeline

This document provides a practical checkpoint timeline from the training history log and groups key checkpoints by behavior regime.

## Checkpoint Cadence
- Target cadence in log: every 500 episodes.
- Visible checkpoint range: ep00500 to ep17000.
- Approximate visible checkpoint count in this range: 34.

## Timeline by Regime

### 1) Early dominance and curriculum climb
Representative checkpoints:
- ep00500
- ep01000
- ep01500

Characteristics:
- rapid curriculum progression to Master
- high robber win-rate
- positive ELO momentum

### 2) First collapse window
Representative checkpoints:
- ep02000
- ep02500

Characteristics:
- robber win-rate approaches 0.00
- sustained negative ELO

### 3) First recovery window
Representative checkpoints:
- ep03000
- ep03500
- ep04000
- ep04500
- ep05000
- ep05500
- ep06000
- ep06500
- ep07000
- ep07500
- ep08000
- ep08500
- ep09000
- ep09500
- ep10000

Characteristics:
- broad recovery to high win-rates
- strong positive ELO, often above +500

### 4) Second collapse window
Representative checkpoints:
- ep10500
- ep11000
- ep11500
- ep12000
- ep12500
- ep13000
- ep13500
- ep14000
- ep14500
- ep15000
- ep15500

Characteristics:
- prolonged 0.00 win-rate region
- deepest negative ELO values

### 5) Final recovery and superior endpoint
Representative checkpoints:
- ep16000
- ep16500
- ep17000

Characteristics:
- win-rate returns to 1.00
- large positive rewards for robber in log
- ELO reaches +1093 by ep17000

## Suggested Evaluation Set
If you want a compact benchmark suite for model comparison, use:
- ep03500 (post-first-recovery baseline)
- ep10500 (collapse stress checkpoint)
- ep15500 (deep collapse checkpoint)
- ep16000 (recovery checkpoint)
- ep17000 (best visible checkpoint)

## Notes
- This timeline is based on the visible training log sequence provided with the project.
- For strict numerical reproducibility, keep the raw history export used for this analysis under versioned artifacts (without secrets).
