# Heist Architect v2 Interactive Checkpoint Workspace

This repository is a dedicated workspace for your Kaggle-trained Heist Architect v2 model artifacts hosted on Hugging Face.

It is focused on three things:
- checkpoint management
- checkpoint validation
- interactive visualization of Architect and Solver behavior

## Visual Preview

![Interactive arena preview](image.png)

## What You Can Do Here

- list all locally available checkpoint episodes
- validate any checkpoint folder (model files + metrics)
- download or resync checkpoints from Hugging Face
- run an interactive browser dashboard to visualize the game loop

## Project Structure

- `main.py`: CLI entry point
- `requirements.txt`: Python dependencies
- `.gitignore`: excludes local secrets, caches, and large artifacts
- `.env`: local runtime configuration and secret values
- `agent.md`: operational guide for coding agents
- `hf_checkpoints/`: local episodes (`epXXXXX`) with model files
- `tools/download_hf_checkpoints.py`: robust multi-pass downloader
- `tools/test_superior_checkpoint.py`: direct checkpoint validator
- `visualization/server.py`: Flask backend for dashboard APIs
- `visualization/static/index.html`: dashboard page
- `visualization/static/style.css`: dashboard styling
- `visualization/static/app.js`: interactive canvas + controls + charts

## Checkpoint Data Contract

Each episode directory under `hf_checkpoints/` should contain:
- `architect.pt`
- `robber.pt`
- `metrics.json`

Example:
- `hf_checkpoints/ep17000/architect.pt`
- `hf_checkpoints/ep17000/robber.pt`
- `hf_checkpoints/ep17000/metrics.json`

## Environment Setup

Use your conda environment:

```bash
conda activate cv_conda
pip install -r requirements.txt
```

## .env Configuration

A local `.env` file is included and auto-loaded by `main.py` at runtime.

Important keys:
- `HF_TOKEN`: required for Hugging Face downloads
- `HEIST_HF_REPO`: defaults to `Shanmuk4622/heist-architect-v2`
- `HEIST_OUT_DIR`: defaults to `hf_checkpoints`
- `HEIST_MAX_PASSES`: number of download retry passes
- `HEIST_RETRY_SLEEP`: wait time between passes

Security note:
- `.env` is git-ignored.
- never commit real tokens to source control.

## CLI Usage

### 1) List local checkpoints

```bash
python main.py list
```

### 2) Validate a checkpoint

```bash
python main.py test --episode ep17000
```

### 3) Download/resync checkpoints

```bash
# Windows PowerShell
$env:HF_TOKEN="YOUR_TOKEN"
python main.py download
```

or use the `.env` value for `HF_TOKEN` and run:

```bash
python main.py download
```

### 4) Launch interactive dashboard

```bash
python main.py dashboard --host 127.0.0.1 --port 5000
```

Open:
- `http://127.0.0.1:5000`

If port `5000` is busy:

```bash
python main.py dashboard --host 127.0.0.1 --port 5002
```

## Interactive Dashboard Features

The website includes:
- live arena canvas replay
- phase visualization:
  - Architect Building (security layout construction)
  - Solver Executing (path traversal under risk)
- playback controls:
  - play
  - pause
  - reset
  - speed slider
  - episode selector
- tactical overlays:
  - solver path
  - camera field-of-view
- telemetry panel:
  - root path
  - episode count
  - latest episode
  - architect steps
  - solver steps
  - current action
- trend charts:
  - robber win rate over episodes
  - architect vs robber ELO

## API Endpoints (Dashboard Backend)

- `GET /api/summary`: workspace and latest checkpoint summary
- `GET /api/episodes`: per-episode metrics and file integrity
- `GET /api/simulate/<episode>`: generated layout + frame-by-frame simulation payload

## Recommended Validation Flow

After any change:

1. run `python main.py list`
2. run `python main.py test --episode ep17000`
3. run `python main.py dashboard --host 127.0.0.1 --port 5000`
4. verify episode switching and playback controls in browser

## Notes

- this workspace intentionally does not include the old full training stack
- this setup is optimized for checkpoint evaluation and presentation
- if you re-download and see missing files, rerun download until `_missing_files.txt` is empty
