"""Dedicated CLI for Kaggle v2 checkpoint management and testing."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch


DEFAULT_REPO = "Shanmuk4622/heist-architect-v2"
DEFAULT_CHECKPOINT_ROOT = Path("hf_checkpoints")


def _load_dotenv(path: Path = Path(".env")):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _episode_dirs(root: Path):
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("ep")])


def cmd_list(args):
    root = Path(args.root)
    episodes = _episode_dirs(root)
    if not episodes:
        print(f"No episode folders found under: {root.resolve()}")
        return

    print(f"Checkpoint root: {root.resolve()}")
    print(f"Episodes found: {len(episodes)}")
    print(f"First: {episodes[0].name}")
    print(f"Latest: {episodes[-1].name}")

    latest = episodes[-1]
    expected = ["architect.pt", "robber.pt", "metrics.json"]
    print("\nLatest files:")
    for name in expected:
        p = latest / name
        print(f"  {'OK ' if p.exists() else 'MISS'} {p}")


def cmd_test(args):
    root = Path(args.root)
    episode_dir = root / args.episode
    arch_path = episode_dir / "architect.pt"
    robber_path = episode_dir / "robber.pt"
    metrics_path = episode_dir / "metrics.json"

    missing = [str(p) for p in [arch_path, robber_path, metrics_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    arch_state = torch.load(arch_path, map_location="cpu")
    robber_state = torch.load(robber_path, map_location="cpu")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    arch_tensors = len(arch_state)
    robber_tensors = len(robber_state)
    arch_params = sum(v.numel() for v in arch_state.values() if hasattr(v, "numel"))
    robber_params = sum(v.numel() for v in robber_state.values() if hasattr(v, "numel"))

    print(f"Episode: {args.episode}")
    print(f"Architect tensors: {arch_tensors} | params: {arch_params:,}")
    print(f"Robber tensors:    {robber_tensors} | params: {robber_params:,}")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def cmd_download(args):
    env = os.environ.copy()
    env["HEIST_HF_REPO"] = args.repo
    env["HEIST_OUT_DIR"] = args.root
    if args.max_passes is not None:
        env["HEIST_MAX_PASSES"] = str(args.max_passes)

    subprocess.run(["python", "tools/download_hf_checkpoints.py"], check=True, env=env)


def cmd_dashboard(args):
    cmd = [
        "python",
        "visualization/server.py",
        "--root",
        args.root,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    subprocess.run(cmd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Heist v2 checkpoint utility")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List downloaded checkpoint episodes")
    p_list.add_argument("--root", default=str(DEFAULT_CHECKPOINT_ROOT))
    p_list.set_defaults(func=cmd_list)

    p_test = sub.add_parser("test", help="Validate a specific checkpoint folder")
    p_test.add_argument("--root", default=str(DEFAULT_CHECKPOINT_ROOT))
    p_test.add_argument("--episode", default="ep17000")
    p_test.set_defaults(func=cmd_test)

    p_dl = sub.add_parser("download", help="Download checkpoints from Hugging Face")
    p_dl.add_argument("--repo", default=DEFAULT_REPO)
    p_dl.add_argument("--root", default=str(DEFAULT_CHECKPOINT_ROOT))
    p_dl.add_argument("--max-passes", type=int, default=8)
    p_dl.set_defaults(func=cmd_download)

    p_dash = sub.add_parser("dashboard", help="Launch checkpoint visualization dashboard")
    p_dash.add_argument("--root", default=str(DEFAULT_CHECKPOINT_ROOT))
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=5000)
    p_dash.set_defaults(func=cmd_dashboard)

    return parser


def main():
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
