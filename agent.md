# Agent Guide: Heist Architect v2 Dedicated Checkpoint Workspace

## Purpose
This repository is dedicated to managing and validating Kaggle-trained Heist Architect v2 checkpoints from Hugging Face.

Primary objectives:
- keep local checkpoint artifacts in `hf_checkpoints/`
- verify checkpoint integrity for a target episode (default: `ep17000`)
- avoid reintroducing legacy training/dashboard code

## Current Workspace Layout
- `main.py`: project CLI (`list`, `test`, `download`)
- `hf_checkpoints/`: episode folders (`epXXXXX`) with model artifacts
- `tools/download_hf_checkpoints.py`: robust Hugging Face downloader with retry passes
- `tools/test_superior_checkpoint.py`: direct checkpoint integrity test script
- `requirements.txt`: minimal dependencies for checkpoint operations
- `kaggle_training/`: reference notebook and notes (read-only context)

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
