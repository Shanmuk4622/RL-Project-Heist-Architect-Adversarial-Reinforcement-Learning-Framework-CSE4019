# Agent Guide: Heist Architect v2 Dedicated Checkpoint Workspace

## Purpose
This repository is dedicated to managing and validating Kaggle-trained Heist Architect v2 checkpoints from Hugging Face.

Primary objectives:
- keep local checkpoint artifacts in `hf_checkpoints/`
- verify checkpoint integrity for a target episode (default: `ep17000`)
- avoid reintroducing legacy training/dashboard code

## Current Workspace Layout
- `main.py`: project CLI (`list`, `test`, `download`, `dashboard`)
- `hf_checkpoints/`: episode folders (`epXXXXX`) with model artifacts
- `tools/download_hf_checkpoints.py`: robust Hugging Face downloader with retry passes
- `tools/test_superior_checkpoint.py`: direct checkpoint integrity test script
- `tools/generate_training_charts.py`: generates training PNG charts from milestone analysis
- `requirements.txt`: minimal dependencies for checkpoint operations
- `kaggle_training/`: reference notebook and notes (read-only context)
- `README_TRAINING_ANALYSIS.md`: long-form training dynamics summary
- `README_TRAINING_CHECKPOINTS.md`: checkpoint timeline and regime grouping
- `README_TRAINING_CHARTS.md`: visual charts report

## Environment
Use this environment before running commands:

```powershell
conda activate cv_conda
```

Install deps if needed:

```powershell
pip install -r requirements.txt
```

## Standard Commands
List local checkpoints:

```powershell
python main.py list
```

Test best checkpoint (default `ep17000`):

```powershell
python main.py test --episode ep17000
```

Launch dashboard:

```powershell
python main.py dashboard --host 127.0.0.1 --port 5000
```

Generate training charts report assets:

```powershell
python tools/generate_training_charts.py
```

Download/sync checkpoints from Hugging Face:

```powershell
$env:HF_TOKEN="<your_token>"
python main.py download
```

Optional direct tools:

```powershell
python tools/test_superior_checkpoint.py --root hf_checkpoints --episode ep17000
python tools/download_hf_checkpoints.py
```

## Data Contract
Each episode folder must contain:
- `architect.pt`
- `robber.pt`
- `metrics.json`

Example:
- `hf_checkpoints/ep17000/architect.pt`
- `hf_checkpoints/ep17000/robber.pt`
- `hf_checkpoints/ep17000/metrics.json`

## Validation Checklist
After any changes, run:

1. `python main.py list`
2. `python main.py test --episode ep17000`

Expected outcome:
- checkpoint root is detected
- latest episode is listed
- both model state dicts load with nonzero parameter counts
- `metrics.json` is readable and printed

## Operational Rules
- Keep this repo focused on checkpoint management and validation.
- Do not add training loops, dashboards, or old legacy modules back unless explicitly requested.
- Prefer using `main.py` commands over ad hoc scripts for routine tasks.
- Keep file/folder names consistent with existing Hugging Face episode layout.

## Hugging Face Notes
- Default model repo is `Shanmuk4622/heist-architect-v2`.
- `HF_TOKEN` is required for download actions.
- If partial download happens, check `hf_checkpoints/_missing_files.txt`.
- Re-run download until missing file count reaches zero.

## Quick Troubleshooting
- `FileNotFoundError` on test:
  - Verify episode exists under `hf_checkpoints/`.
  - Re-run download sync.
- `HF_TOKEN env var is required`:
  - Set `HF_TOKEN` in current shell.
- Slow/failed downloads:
  - Re-run `python main.py download`; downloader uses multiple retry passes.

## Definition of Done for Agent Tasks
A task is complete only when:
- required file edits are applied
- checkpoint test command succeeds (`python main.py test --episode ep17000`)
- output confirms model files and metrics load correctly

## How To Use This agent.md
Use this file as your operating checklist when working in this repo:

1. Start by activating `cv_conda` and checking `main.py list`.
2. For model integrity work, run `main.py test --episode ep17000` before and after edits.
3. For dashboard work, launch `main.py dashboard` and validate visual behavior.
4. For training-history documentation updates, regenerate charts with `tools/generate_training_charts.py` and update README links.
5. Before finishing, ensure no secrets are committed and checkpoint blobs remain untracked by git.
