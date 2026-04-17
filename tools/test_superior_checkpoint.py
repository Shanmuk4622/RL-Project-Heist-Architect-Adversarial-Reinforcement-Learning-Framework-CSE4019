import argparse
import json
from pathlib import Path

import torch


def run_test(root: Path, episode: str):
    episode_dir = root / episode
    arch_path = episode_dir / "architect.pt"
    robber_path = episode_dir / "robber.pt"
    metrics_path = episode_dir / "metrics.json"

    for p in [arch_path, robber_path, metrics_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    arch_state = torch.load(arch_path, map_location="cpu")
    robber_state = torch.load(robber_path, map_location="cpu")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    arch_params = sum(v.numel() for v in arch_state.values() if hasattr(v, "numel"))
    robber_params = sum(v.numel() for v in robber_state.values() if hasattr(v, "numel"))

    print(f"Checkpoint: {episode_dir}")
    print(f"Architect keys: {len(arch_state)}, params: {arch_params:,}")
    print(f"Robber keys: {len(robber_state)}, params: {robber_params:,}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate superior Heist v2 checkpoint")
    parser.add_argument("--root", default="hf_checkpoints")
    parser.add_argument("--episode", default="ep17000")
    args = parser.parse_args()
    run_test(Path(args.root), args.episode)
