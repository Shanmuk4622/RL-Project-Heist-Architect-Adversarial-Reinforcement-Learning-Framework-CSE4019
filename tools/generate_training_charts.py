"""Generate PNG charts from training milestone data in README_TRAINING_ANALYSIS.md."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_MD = ROOT / "README_TRAINING_ANALYSIS.md"
OUT_DIR = ROOT / "reports" / "assets"


def parse_milestones(md_text: str):
    points = []
    # Parse markdown table rows: | 50 | Rookie | 0.96 | +477 | ... |
    row_re = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*([+-]?\d+)\s*\|",
        re.MULTILINE,
    )
    for m in row_re.finditer(md_text):
        points.append(
            {
                "episode": int(m.group(1)),
                "stage": m.group(2).strip(),
                "robber_win": float(m.group(3)),
                "elo_diff": int(m.group(4)),
            }
        )
    points.sort(key=lambda x: x["episode"])
    return points


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_winrate_chart(points):
    x = [p["episode"] for p in points]
    y = [p["robber_win"] for p in points]

    plt.figure(figsize=(12, 5.5), dpi=150)
    plt.plot(x, y, marker="o", linewidth=2.0, color="#14b8a6")
    plt.title("Robber Win Rate vs Episode (Milestones)")
    plt.xlabel("Episode")
    plt.ylabel("Robber Win Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)

    for p in points:
        if p["episode"] in (50, 1750, 3000, 10200, 15500, 17000):
            plt.annotate(
                f"ep{p['episode']}\n{p['robber_win']:.2f}",
                (p["episode"], p["robber_win"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    out = OUT_DIR / "training_winrate_milestones.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def save_elo_chart(points):
    x = [p["episode"] for p in points]
    y = [p["elo_diff"] for p in points]

    plt.figure(figsize=(12, 5.5), dpi=150)
    plt.plot(x, y, marker="o", linewidth=2.0, color="#2563eb")
    plt.axhline(0, linestyle="--", linewidth=1.1, color="#ef4444", alpha=0.75)
    plt.title("ELO Diff vs Episode (Milestones)")
    plt.xlabel("Episode")
    plt.ylabel("ELO Diff (Robber - Architect)")
    plt.grid(alpha=0.25)

    for p in points:
        if p["episode"] in (50, 1750, 2950, 3000, 5350, 15500, 17000):
            plt.annotate(
                f"ep{p['episode']}\n{p['elo_diff']:+d}",
                (p["episode"], p["elo_diff"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    out = OUT_DIR / "training_elo_milestones.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def main():
    if not ANALYSIS_MD.exists():
        raise FileNotFoundError(f"Missing analysis file: {ANALYSIS_MD}")

    text = ANALYSIS_MD.read_text(encoding="utf-8")
    points = parse_milestones(text)
    if not points:
        raise RuntimeError("No milestone rows found in README_TRAINING_ANALYSIS.md")

    ensure_out_dir()
    win_path = save_winrate_chart(points)
    elo_path = save_elo_chart(points)

    print(f"Generated: {win_path}")
    print(f"Generated: {elo_path}")


if __name__ == "__main__":
    main()
