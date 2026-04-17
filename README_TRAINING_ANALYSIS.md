# Training Analysis Report (from history log)

## Scope
This report summarizes the long-form training history that was provided in the shared training log.

Covered range:
- earliest visible milestone: episode 50
- latest visible milestone: episode 17000
- curriculum levels observed: Rookie, Intermediate, Expert, Master
- checkpoint cadence: every 500 episodes

## Executive Summary
The training history shows a repeated adversarial dynamic with three major patterns:

1. Fast solver dominance in early curriculum.
2. Hard collapses where solver win-rate drops to 0.00 for long windows.
3. Strong recoveries after resume/restart, ending at a best visible state near episode 17000.

Final visible state (episode 17000):
- stage: Master
- robber_win: 1.00
- robber reward: +2.3
- architect reward: -2.3
- ELO diff: +1093

## Key Milestones

| Episode | Stage | Robber Win | ELO Diff | Interpretation |
|---:|---|---:|---:|---|
| 50 | Rookie | 0.96 | +477 | Solver starts very strong |
| 100 | Intermediate | 0.98 | +609 | Early curriculum success |
| 200 | Expert | 1.00 | +737 | Solver adapts quickly |
| 300 | Master | 1.00 | +810 | Reaches hardest curriculum quickly |
| 1750 | Master | 0.59 | -381 | First major collapse begins |
| 2000 | Master | 0.00 | -798 | Collapse deepens |
| 2950 | Master | 0.00 | -1060 | Worst region of first collapse |
| 3000 | Master | 0.28 | +206 | Recovery starts |
| 3500 | Master | 1.00 | +731 | Recovery stabilizes |
| 5350 | Master | 1.00 | +994 | Peak before second collapse window |
| 10200 | Master | 0.55 | -382 | Second collapse begins |
| 10500 | Master | 0.00 | -828 | Second collapse deepens |
| 15500 | Master | 0.00 | -1317 | Deepest visible valley |
| 16000 | Master | 1.00 | +895 | Strong recovery after 30000-episode restart |
| 17000 | Master | 1.00 | +1093 | Best visible endpoint |

## Phase-by-Phase Behavior

### Phase A: Curriculum acceleration (50 to 1700)
- Rapid movement from Rookie to Master.
- Win-rate mostly 0.90 to 1.00.
- ELO diff rises to roughly +684 before instability appears.

### Phase B: First collapse (1750 to 2950)
- Win-rate degrades to 0.00 for many checkpoints.
- ELO diff crosses negative and continues to around -1060.
- Indicates architect policy temporarily overpowers solver policy in the active regime.

### Phase C: First recovery and expansion (3000 to 10000)
- Win-rate recovers from 0.28 to 0.90+ and often 1.00.
- ELO diff climbs back positive and reaches a visible high near +994.
- Long stable stretch with frequent successful HF checkpoints.

### Phase D: Second collapse (10200 to 15500)
- Another prolonged negative regime.
- Win-rate stays near 0.00 for a large span.
- ELO diff reaches deepest visible value around -1317.

### Phase E: Recovery after restart (15501 to 17000)
- Training restarts with 30000 target episodes and resumes from ep16500 later.
- Win-rate returns to 1.00 consistently.
- Reward magnitude increases (rob_r up to +2.3), and ELO climbs to +1093 by ep17000.

## Checkpoint Throughput Observations
- Checkpoints are uploaded every 500 episodes.
- Typical upload logs show two-file processing with totals near 62.2MB and new upload near 52MB to 62MB.
- The visible checkpoint sequence reaches ep17000.

## What This Means for the Current Model
The visible endpoint suggests the final published checkpoint at ep17000 is from a high-performance recovery regime, not from a collapse regime. This supports using ep17000 as the default "superior checkpoint" for testing and dashboard demonstrations.

## Practical Recommendation
1. Keep ep17000 as the default evaluation checkpoint.
2. Preserve at least one checkpoint from each regime for regression testing:
   - pre-collapse strong: ep1700 or ep3500
   - collapse: ep10500 or ep15500
   - recovery: ep16000 and ep17000
3. Add scenario-based tests in the dashboard that compare solver behavior across these regimes.
