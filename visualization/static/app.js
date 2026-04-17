let winRateChart;
let eloChart;
let simData;
let isPlaying = false;
let playIndex = 0;
let lastTick = 0;
let pxPerCell = 1;

const state = {
  speed: 1,
  showPath: true,
  showFov: true,
};

const canvas = document.getElementById("arenaCanvas");
const ctx = canvas.getContext("2d");
const episodeSelect = document.getElementById("episodeSelect");
const speedRange = document.getElementById("speedRange");

function pickMetric(metrics, key) {
  const value = metrics?.[key];
  return typeof value === "number" ? value : null;
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function fitCanvas() {
  const size = Math.floor(canvas.clientWidth);
  if (!size || size === canvas.width) {
    return;
  }
  canvas.width = size;
  canvas.height = size;
  draw();
}

function drawGrid(size) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  pxPerCell = canvas.width / size;

  ctx.fillStyle = "#10204a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(76, 133, 204, 0.17)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= size; i += 1) {
    const p = i * pxPerCell;
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(canvas.width, p);
    ctx.stroke();
  }
}

function fillCell(r, c, color) {
  const pad = pxPerCell * 0.08;
  ctx.fillStyle = color;
  ctx.fillRect(c * pxPerCell + pad, r * pxPerCell + pad, pxPerCell - pad * 2, pxPerCell - pad * 2);
}

function drawCircle(r, c, color, radiusScale = 0.3) {
  const cx = (c + 0.5) * pxPerCell;
  const cy = (r + 0.5) * pxPerCell;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, pxPerCell * radiusScale, 0, Math.PI * 2);
  ctx.fill();
}

function drawFov(camera, heading, range) {
  const [r, c] = camera.at;
  let dr = 0;
  let dc = 0;
  if (heading === 0) {
    dr = 1;
  } else if (heading === 1) {
    dr = -1;
  } else if (heading === 2) {
    dc = 1;
  } else {
    dc = -1;
  }

  ctx.fillStyle = "rgba(255, 111, 137, 0.18)";
  for (let i = 1; i <= range; i += 1) {
    fillCell(r + dr * i, c + dc * i, "rgba(255, 111, 137, 0.18)");
  }
}

function draw() {
  if (!simData) {
    return;
  }

  const { layout, frames } = simData;
  const buildSteps = layout.build_order.length;
  const frame = frames[Math.max(0, Math.min(playIndex - buildSteps, frames.length - 1))] || null;
  const size = layout.size;

  drawGrid(size);

  const builtCount = Math.min(playIndex, buildSteps);
  const builtWalls = new Set();
  const builtCameras = new Set();
  const builtGuards = new Set();
  for (let i = 0; i < builtCount; i += 1) {
    const step = layout.build_order[i];
    if (step.type === "wall") {
      builtWalls.add(step.at.join(","));
    } else if (step.type === "camera") {
      builtCameras.add(step.id);
    } else if (step.type === "guard") {
      builtGuards.add(step.id);
    }
  }

  for (const wall of layout.walls) {
    const key = wall.join(",");
    if (builtWalls.has(key) || playIndex >= buildSteps) {
      fillCell(wall[0], wall[1], "#6f829f");
    }
  }

  fillCell(layout.start[0], layout.start[1], "#32d99f");
  drawCircle(layout.vault[0], layout.vault[1], "#ffc746", 0.34);

  if (playIndex >= buildSteps) {
    const cameraStateById = new Map((frame?.camera_states || []).map((x) => [x.id, x.heading]));
    for (const cam of layout.cameras) {
      if (!builtCameras.has(cam.id) && playIndex < buildSteps) {
        continue;
      }
      const heading = cameraStateById.has(cam.id) ? cameraStateById.get(cam.id) : cam.heading;
      if (state.showFov) {
        drawFov(cam, heading, cam.range);
      }
      drawCircle(cam.at[0], cam.at[1], "#ff6f89", 0.24);
    }

    const guardStateById = new Map((frame?.guard_states || []).map((x) => [x.id, x.at]));
    for (const g of layout.guards) {
      if (!builtGuards.has(g.id) && playIndex < buildSteps) {
        continue;
      }
      const at = guardStateById.get(g.id) || g.patrol[0];
      drawCircle(at[0], at[1], "#ffb347", 0.22);
    }

    if (state.showPath) {
      ctx.strokeStyle = "rgba(45, 227, 255, 0.75)";
      ctx.lineWidth = Math.max(2, pxPerCell * 0.08);
      ctx.beginPath();
      let hasPoint = false;
      for (let i = 0; i <= Math.max(0, playIndex - buildSteps); i += 1) {
        const f = frames[i];
        if (!f) {
          continue;
        }
        const x = (f.solver[1] + 0.5) * pxPerCell;
        const y = (f.solver[0] + 0.5) * pxPerCell;
        if (!hasPoint) {
          ctx.moveTo(x, y);
          hasPoint = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    if (frame) {
      const color = frame.detected ? "#ff3b57" : "#2de3ff";
      drawCircle(frame.solver[0], frame.solver[1], color, 0.26);
    }
  }

  const phase = playIndex < buildSteps ? "Architect Building" : "Solver Executing";
  setText("phasePill", `Phase: ${phase}`);
  setText("stepLabel", `Step ${playIndex + 1}/${buildSteps + frames.length}`);
  setText("architectSteps", String(buildSteps));
  setText("solverSteps", String(frames.length));

  if (frame) {
    const actionLabel = frame.detected ? `${frame.action.toUpperCase()} (detected)` : frame.action.toUpperCase();
    setText("currentAction", actionLabel);
  } else {
    setText("currentAction", "BUILDING");
  }
}

function destroyIfExists(chart) {
  if (chart) {
    chart.destroy();
  }
}

function renderCharts(episodes) {
  const sorted = episodes.slice().sort((a, b) => a.episode_num - b.episode_num);
  const labels = sorted.map((e) => e.episode);

  const winRateData = sorted.map((e) => pickMetric(e.metrics, "robber_win_rate"));
  const archElo = sorted.map((e) => pickMetric(e.metrics, "arch_elo"));
  const robberElo = sorted.map((e) => pickMetric(e.metrics, "robber_elo"));

  destroyIfExists(winRateChart);
  destroyIfExists(eloChart);

  winRateChart = new Chart(document.getElementById("winRateChart"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Robber Win Rate",
        data: winRateData,
        borderColor: "#5af79c",
        backgroundColor: "rgba(90,247,156,0.1)",
        tension: 0.24,
        pointRadius: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#98b4d8", maxTicksLimit: 12 } },
        y: { ticks: { color: "#98b4d8" }, min: 0, max: 1 },
      },
      plugins: { legend: { labels: { color: "#e8f2ff" } } },
    },
  });

  eloChart = new Chart(document.getElementById("eloChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Architect ELO",
          data: archElo,
          borderColor: "#2de3ff",
          pointRadius: 2,
          tension: 0.2,
        },
        {
          label: "Robber ELO",
          data: robberElo,
          borderColor: "#ffc746",
          pointRadius: 2,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#98b4d8", maxTicksLimit: 12 } },
        y: { ticks: { color: "#98b4d8" } },
      },
      plugins: { legend: { labels: { color: "#e8f2ff" } } },
    },
  });
}

async function loadSimulation(episode) {
  const res = await fetch(`/api/simulate/${episode}`);
  simData = await res.json();
  playIndex = 0;
  draw();
}

function tick(ts) {
  if (!isPlaying || !simData) {
    requestAnimationFrame(tick);
    return;
  }

  const total = simData.layout.build_order.length + simData.frames.length;
  const stepMs = 220 / state.speed;
  if (ts - lastTick >= stepMs) {
    playIndex += 1;
    if (playIndex >= total) {
      playIndex = total - 1;
      isPlaying = false;
    }
    draw();
    lastTick = ts;
  }

  requestAnimationFrame(tick);
}

async function bootstrap() {
  const [summaryRes, episodesRes] = await Promise.all([fetch("/api/summary"), fetch("/api/episodes")]);
  const summary = await summaryRes.json();
  const episodes = (await episodesRes.json()).episodes || [];

  setText("rootPath", summary.root ?? "-");
  setText("episodeCount", String(summary.count ?? 0));
  setText("latestEpisode", summary.latest ?? "-");

  const sorted = episodes.slice().sort((a, b) => a.episode_num - b.episode_num);
  for (const ep of sorted) {
    const opt = document.createElement("option");
    opt.value = ep.episode;
    opt.textContent = ep.episode;
    episodeSelect.appendChild(opt);
  }

  renderCharts(episodes);

  const initial = summary.latest || sorted.at(-1)?.episode;
  if (initial) {
    episodeSelect.value = initial;
    await loadSimulation(initial);
  }
}

function wireEvents() {
  document.getElementById("playBtn").addEventListener("click", () => {
    isPlaying = true;
  });

  document.getElementById("pauseBtn").addEventListener("click", () => {
    isPlaying = false;
  });

  document.getElementById("resetBtn").addEventListener("click", () => {
    isPlaying = false;
    playIndex = 0;
    draw();
  });

  episodeSelect.addEventListener("change", async (e) => {
    isPlaying = false;
    await loadSimulation(e.target.value);
  });

  speedRange.addEventListener("input", () => {
    state.speed = Number(speedRange.value);
    setText("speedValue", `${state.speed.toFixed(2)}x`);
  });

  document.getElementById("showPath").addEventListener("change", (e) => {
    state.showPath = e.target.checked;
    draw();
  });

  document.getElementById("showFov").addEventListener("change", (e) => {
    state.showFov = e.target.checked;
    draw();
  });

  window.addEventListener("resize", fitCanvas);
}

wireEvents();
bootstrap()
  .then(() => {
    fitCanvas();
    requestAnimationFrame(tick);
  })
  .catch((err) => {
    console.error(err);
    alert(`Failed to load dashboard: ${err}`);
  });
