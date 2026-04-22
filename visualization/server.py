"""Interactive checkpoint dashboard server for Heist Architect v2 artifacts."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, send_from_directory


app = Flask(__name__, static_folder="static")
CHECKPOINT_ROOT = Path("hf_checkpoints")
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _episode_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("ep")])


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_episode_number(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1


def _build_point(episode_dir: Path) -> dict:
    episode = episode_dir.name
    metrics_path = episode_dir / "metrics.json"
    arch_path = episode_dir / "architect.pt"
    robber_path = episode_dir / "robber.pt"

    point = {
        "episode": episode,
        "episode_num": _extract_episode_number(episode),
        "has_architect": arch_path.exists(),
        "has_robber": robber_path.exists(),
        "has_metrics": metrics_path.exists(),
        "metrics": {},
    }

    if metrics_path.exists():
        try:
            point["metrics"] = _read_json(metrics_path)
        except Exception as ex:  # pragma: no cover
            point["metrics_error"] = str(ex)

    return point


def _in_bounds(pos: tuple[int, int], size: int) -> bool:
    r, c = pos
    return 0 <= r < size and 0 <= c < size


def _neighbors(pos: tuple[int, int], size: int):
    for dr, dc in DIRS:
        nxt = (pos[0] + dr, pos[1] + dc)
        if _in_bounds(nxt, size):
            yield nxt


def _bfs_path(start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]], size: int):
    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in _neighbors(cur, size):
            if nxt in blocked or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)

    if goal not in parent:
        return []

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _generate_layout(seed: int, size: int = 16):
    rng = random.Random(seed)
    start = (1, 1)
    vault = (size - 2, size - 2)

    blocked = set()
    for i in range(size):
        blocked.add((0, i))
        blocked.add((size - 1, i))
        blocked.add((i, 0))
        blocked.add((i, size - 1))

    build_order = []
    candidates = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    rng.shuffle(candidates)

    for r, c in candidates:
        if (r, c) in (start, vault):
            continue
        if rng.random() > 0.2:
            continue

        blocked.add((r, c))
        path = _bfs_path(start, vault, blocked, size)
        if not path:
            blocked.remove((r, c))
        else:
            build_order.append({"type": "wall", "at": [r, c]})

    free_cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1) if (r, c) not in blocked and (r, c) not in (start, vault)]
    rng.shuffle(free_cells)

    cameras = []
    for idx, cell in enumerate(free_cells[:4]):
        camera = {
            "id": idx,
            "at": [cell[0], cell[1]],
            "heading": rng.choice([0, 1, 2, 3]),
            "sweep": rng.choice([8, 10, 12]),
            "range": rng.choice([4, 5]),
        }
        cameras.append(camera)
        build_order.append({"type": "camera", "id": idx, "at": [cell[0], cell[1]]})

    guards = []
    for idx, cell in enumerate(free_cells[4:6]):
        r, c = cell
        patrol = [[r, c], [max(1, r - 2), c], [r, min(size - 2, c + 2)], [min(size - 2, r + 1), c]]
        guard = {
            "id": idx,
            "patrol": patrol,
            "cycle": 16,
        }
        guards.append(guard)
        build_order.append({"type": "guard", "id": idx, "at": [r, c]})

    return {
        "size": size,
        "start": list(start),
        "vault": list(vault),
        "walls": [list(x) for x in sorted(blocked)],
        "cameras": cameras,
        "guards": guards,
        "build_order": build_order,
    }


def _camera_heading(camera: dict, t: int) -> int:
    base = camera["heading"]
    sweep = max(2, int(camera["sweep"]))
    step = (t // (sweep // 2)) % 4
    return (base + step) % 4


def _camera_sees(camera: dict, target: tuple[int, int], walls: set[tuple[int, int]], t: int) -> bool:
    r, c = camera["at"]
    heading = _camera_heading(camera, t)
    dr, dc = DIRS[heading]
    cur = (r, c)
    max_len = int(camera["range"])
    for _ in range(max_len):
        cur = (cur[0] + dr, cur[1] + dc)
        if cur in walls:
            return False
        if cur == target:
            return True
    return False


def _guard_pos(guard: dict, t: int) -> tuple[int, int]:
    patrol = guard["patrol"]
    if not patrol:
        return (0, 0)
    idx = t % len(patrol)
    r, c = patrol[idx]
    return (r, c)


def _simulate_solver(layout: dict):
    size = int(layout["size"])
    walls = {tuple(x) for x in layout["walls"]}
    start = tuple(layout["start"])
    vault = tuple(layout["vault"])
    path = _bfs_path(start, vault, walls, size)
    if not path:
        return []

    frames = []
    t = 0
    step_i = 0
    last = path[0]
    wait_budget = 3

    while step_i < len(path):
        target = path[step_i]
        dangerous = False

        for cam in layout["cameras"]:
            if _camera_sees(cam, target, walls, t):
                dangerous = True
                break

        if not dangerous:
            for guard in layout["guards"]:
                if _guard_pos(guard, t) == target:
                    dangerous = True
                    break

        action = "move"
        if dangerous and wait_budget > 0 and target != vault:
            action = "wait"
            solver_pos = last
            wait_budget -= 1
        else:
            solver_pos = target
            step_i += 1
            wait_budget = 3
            last = solver_pos

        frames.append(
            {
                "t": t,
                "solver": [solver_pos[0], solver_pos[1]],
                "action": action,
                "detected": dangerous and action == "move",
                "camera_states": [{"id": cam["id"], "heading": _camera_heading(cam, t)} for cam in layout["cameras"]],
                "guard_states": [{"id": g["id"], "at": list(_guard_pos(g, t))} for g in layout["guards"]],
            }
        )

        t += 1
        if t > 500:
            break

    return frames


def _build_simulation(episode: str):
    seed = _extract_episode_number(episode)
    if seed < 0:
        seed = 17000
    layout = _generate_layout(seed)
    frames = _simulate_solver(layout)
    return {
        "episode": episode,
        "layout": layout,
        "frames": frames,
        "timeline": {
            "architect_steps": len(layout["build_order"]),
            "solver_steps": len(frames),
        },
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/summary")
def api_summary():
    episodes = _episode_dirs(CHECKPOINT_ROOT)
    if not episodes:
        return jsonify(
            {
                "root": str(CHECKPOINT_ROOT.resolve()),
                "count": 0,
                "first": None,
                "latest": None,
                "latest_files": {},
            }
        )

    latest = episodes[-1]
    latest_files = {
        "architect.pt": (latest / "architect.pt").exists(),
        "robber.pt": (latest / "robber.pt").exists(),
        "metrics.json": (latest / "metrics.json").exists(),
    }

    return jsonify(
        {
            "root": str(CHECKPOINT_ROOT.resolve()),
            "count": len(episodes),
            "first": episodes[0].name,
            "latest": latest.name,
            "latest_files": latest_files,
        }
    )


@app.route("/api/episodes")
def api_episodes():
    episodes = _episode_dirs(CHECKPOINT_ROOT)
    data = [_build_point(ep) for ep in episodes]
    return jsonify({"episodes": data})


@app.route("/api/simulate/<episode>")
def api_simulate(episode: str):
    return jsonify(_build_simulation(episode))

import threading
import webbrowser

def main():
    parser = argparse.ArgumentParser(description="Heist v2 dashboard server")
    parser.add_argument("--root", default="hf_checkpoints")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    global CHECKPOINT_ROOT
    CHECKPOINT_ROOT = Path(args.root)

    # Auto-open browser after a short delay
    url = f"http://{args.host}:{args.port}"
    print(f"Opening dashboard at {url} ...")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
