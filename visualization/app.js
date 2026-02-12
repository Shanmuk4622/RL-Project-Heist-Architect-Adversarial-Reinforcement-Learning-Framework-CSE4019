/**
 * Heist Architect — Visualization Dashboard
 * Canvas-based 2D renderer + chart system + WebSocket integration
 * BRIGHT THEME — vibrant color palette
 */

// ============================================================
// Configuration — Bright Color Palette
// ============================================================

const CONFIG = {
    TILE_COLORS: {
        0: '#e8ecf4',   // Empty  (soft lavender-gray)
        1: '#6b7280',   // Wall   (cool gray)
        2: '#10b981',   // Start  (emerald green)
        3: '#f43f5e',   // Vault  (rose red)
        4: '#8b5cf6',   // Camera (violet purple)
        5: '#f97316',   // Guard  (vibrant orange)
    },
    WALL_ACCENT: '#9ca3af',
    VISION_COLOR: 'rgba(244, 63, 94, 0.14)',
    VISION_BORDER: 'rgba(244, 63, 94, 0.28)',
    SOLVER_COLOR: '#eab308',
    SOLVER_GLOW_INNER: 'rgba(234, 179, 8, 0.35)',
    SOLVER_GLOW_OUTER: 'rgba(234, 179, 8, 0)',
    SOLVER_TRAIL_COLOR: 'rgba(234, 179, 8, 0.20)',
    GRID_LINE_COLOR: 'rgba(100, 116, 180, 0.08)',
    CAMERA_CONE_COLOR: 'rgba(139, 92, 246, 0.18)',
    CAMERA_BODY: '#8b5cf6',
    CAMERA_DIR: '#a78bfa',
    GUARD_PATROL_COLOR: 'rgba(249, 115, 22, 0.18)',
    GUARD_BODY: '#f97316',
    GUARD_DIR: '#fb923c',
    CANVAS_BG: '#e8ecf4',
    CHART_BG: 'rgba(248, 249, 253, 0.95)',
    CHART_GRID: 'rgba(100, 116, 180, 0.08)',
    CHART_LABEL: 'rgba(90, 95, 128, 0.6)',
    CHART_LEGEND: 'rgba(90, 95, 128, 0.75)',
    DETECTION_X: '#ef4444',
};

// ============================================================
// State
// ============================================================

const state = {
    envState: null,
    metrics: {
        episode: [], solve_rate: [], detection_rate: [],
        architect_reward: [], solver_reward: []
    },
    showVisibility: true,
    activeChart: 'rewards',
    isTraining: false,
    totalEpisodes: 500,
    animFrame: 0,
};

// ============================================================
// Canvas Setup
// ============================================================

const gridCanvas = document.getElementById('gridCanvas');
const gridCtx = gridCanvas.getContext('2d');
const chartCanvas = document.getElementById('chartCanvas');
const chartCtx = chartCanvas.getContext('2d');

function resizeCanvas() {
    const container = gridCanvas.parentElement;
    const size = Math.min(container.clientWidth - 40, 600);
    gridCanvas.width = size;
    gridCanvas.height = size;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ============================================================
// Grid Renderer
// ============================================================

function renderGrid(envState) {
    if (!envState || !envState.grid) return;

    const grid = envState.grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const tileW = gridCanvas.width / cols;
    const tileH = gridCanvas.height / rows;

    // Clear with bright bg
    gridCtx.fillStyle = CONFIG.CANVAS_BG;
    gridCtx.fillRect(0, 0, gridCanvas.width, gridCanvas.height);

    // Draw tiles
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const tile = grid[r][c];
            if (tile !== 0) {
                gridCtx.fillStyle = CONFIG.TILE_COLORS[tile] || '#aaa';
                gridCtx.fillRect(c * tileW, r * tileH, tileW, tileH);

                // Add highlight for walls (slight 3D effect)
                if (tile === 1) {
                    gridCtx.fillStyle = 'rgba(255,255,255,0.12)';
                    gridCtx.fillRect(c * tileW, r * tileH, tileW, 1.5);
                    gridCtx.fillRect(c * tileW, r * tileH, 1.5, tileH);
                    gridCtx.fillStyle = 'rgba(0,0,0,0.06)';
                    gridCtx.fillRect(c * tileW, r * tileH + tileH - 1.5, tileW, 1.5);
                    gridCtx.fillRect(c * tileW + tileW - 1.5, r * tileH, 1.5, tileH);
                }
            }
        }
    }

    // Draw visibility map
    if (state.showVisibility && envState.visibility) {
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                if (envState.visibility[r][c] > 0.5) {
                    gridCtx.fillStyle = CONFIG.VISION_COLOR;
                    gridCtx.fillRect(c * tileW, r * tileH, tileW, tileH);

                    gridCtx.strokeStyle = CONFIG.VISION_BORDER;
                    gridCtx.lineWidth = 0.5;
                    gridCtx.strokeRect(c * tileW, r * tileH, tileW, tileH);
                }
            }
        }
    }

    // Draw camera vision cones
    if (envState.cameras) {
        envState.cameras.forEach(cam => {
            drawCameraCone(cam, tileW, tileH);
        });
    }

    // Draw guard patrol paths
    if (envState.guards) {
        envState.guards.forEach(guard => {
            drawGuardPatrol(guard, tileW, tileH);
        });
    }

    // Draw solver trail
    if (envState.solver_path && envState.solver_path.length > 1) {
        gridCtx.strokeStyle = CONFIG.SOLVER_TRAIL_COLOR;
        gridCtx.lineWidth = 2.5;
        gridCtx.lineJoin = 'round';
        gridCtx.lineCap = 'round';
        gridCtx.beginPath();
        envState.solver_path.forEach((pos, i) => {
            const x = pos[1] * tileW + tileW / 2;
            const y = pos[0] * tileH + tileH / 2;
            if (i === 0) gridCtx.moveTo(x, y);
            else gridCtx.lineTo(x, y);
        });
        gridCtx.stroke();
    }

    // Draw solver position
    if (envState.solver_pos) {
        const [sr, sc] = envState.solver_pos;
        const cx = sc * tileW + tileW / 2;
        const cy = sr * tileH + tileH / 2;
        const radius = Math.min(tileW, tileH) * 0.35;

        // Glow
        const glow = gridCtx.createRadialGradient(cx, cy, 0, cx, cy, radius * 2.5);
        glow.addColorStop(0, CONFIG.SOLVER_GLOW_INNER);
        glow.addColorStop(1, CONFIG.SOLVER_GLOW_OUTER);
        gridCtx.fillStyle = glow;
        gridCtx.beginPath();
        gridCtx.arc(cx, cy, radius * 2.5, 0, Math.PI * 2);
        gridCtx.fill();

        // Solver dot
        gridCtx.fillStyle = CONFIG.SOLVER_COLOR;
        gridCtx.beginPath();
        gridCtx.arc(cx, cy, radius, 0, Math.PI * 2);
        gridCtx.fill();

        // Border
        gridCtx.strokeStyle = 'rgba(0,0,0,0.15)';
        gridCtx.lineWidth = 1.5;
        gridCtx.stroke();

        // Inner shine
        gridCtx.fillStyle = 'rgba(255, 255, 255, 0.45)';
        gridCtx.beginPath();
        gridCtx.arc(cx - radius * 0.2, cy - radius * 0.25, radius * 0.35, 0, Math.PI * 2);
        gridCtx.fill();
    }

    // Draw start/vault labels
    if (envState.start_pos) {
        drawLabel('S', envState.start_pos, tileW, tileH, '#fff');
    }
    if (envState.vault_pos) {
        drawLabel('V', envState.vault_pos, tileW, tileH, '#fff');
    }

    // Grid lines
    gridCtx.strokeStyle = CONFIG.GRID_LINE_COLOR;
    gridCtx.lineWidth = 0.5;
    for (let r = 0; r <= rows; r++) {
        gridCtx.beginPath();
        gridCtx.moveTo(0, r * tileH);
        gridCtx.lineTo(gridCanvas.width, r * tileH);
        gridCtx.stroke();
    }
    for (let c = 0; c <= cols; c++) {
        gridCtx.beginPath();
        gridCtx.moveTo(c * tileW, 0);
        gridCtx.lineTo(c * tileW, gridCanvas.height);
        gridCtx.stroke();
    }

    // Detection events — red X marks
    if (envState.detection_events) {
        envState.detection_events.forEach(evt => {
            const [dr, dc] = evt.position;
            const x = dc * tileW + tileW / 2;
            const y = dr * tileH + tileH / 2;

            gridCtx.strokeStyle = CONFIG.DETECTION_X;
            gridCtx.lineWidth = 2.5;
            gridCtx.lineCap = 'round';
            const s = tileW * 0.35;
            gridCtx.beginPath();
            gridCtx.moveTo(x - s, y - s);
            gridCtx.lineTo(x + s, y + s);
            gridCtx.moveTo(x + s, y - s);
            gridCtx.lineTo(x - s, y + s);
            gridCtx.stroke();
        });
    }
}

function drawCameraCone(cam, tileW, tileH) {
    const cx = cam.col * tileW + tileW / 2;
    const cy = cam.row * tileH + tileH / 2;
    const range = (cam.vision_range || 6) * tileW;
    const heading = -cam.heading * Math.PI / 180;
    const halfFov = (cam.fov_angle || 60) * Math.PI / 360;

    // Cone fill
    gridCtx.fillStyle = CONFIG.CAMERA_CONE_COLOR;
    gridCtx.beginPath();
    gridCtx.moveTo(cx, cy);
    gridCtx.arc(cx, cy, range, heading - halfFov, heading + halfFov);
    gridCtx.closePath();
    gridCtx.fill();

    // Camera body
    gridCtx.fillStyle = CONFIG.CAMERA_BODY;
    gridCtx.beginPath();
    gridCtx.arc(cx, cy, tileW * 0.28, 0, Math.PI * 2);
    gridCtx.fill();
    gridCtx.strokeStyle = 'rgba(0,0,0,0.12)';
    gridCtx.lineWidth = 1;
    gridCtx.stroke();

    // Direction indicator
    gridCtx.strokeStyle = CONFIG.CAMERA_DIR;
    gridCtx.lineWidth = 2;
    gridCtx.lineCap = 'round';
    gridCtx.beginPath();
    gridCtx.moveTo(cx, cy);
    gridCtx.lineTo(cx + Math.cos(heading) * tileW * 0.5,
        cy + Math.sin(heading) * tileH * 0.5);
    gridCtx.stroke();
}

function drawGuardPatrol(guard, tileW, tileH) {
    if (!guard.patrol_path || guard.patrol_path.length < 2) return;

    // Draw patrol path
    gridCtx.strokeStyle = CONFIG.GUARD_PATROL_COLOR;
    gridCtx.lineWidth = 1.5;
    gridCtx.setLineDash([5, 5]);
    gridCtx.lineCap = 'round';
    gridCtx.beginPath();
    guard.patrol_path.forEach((pos, i) => {
        const x = pos[1] * tileW + tileW / 2;
        const y = pos[0] * tileH + tileH / 2;
        if (i === 0) gridCtx.moveTo(x, y);
        else gridCtx.lineTo(x, y);
    });
    const first = guard.patrol_path[0];
    gridCtx.lineTo(first[1] * tileW + tileW / 2, first[0] * tileH + tileH / 2);
    gridCtx.stroke();
    gridCtx.setLineDash([]);

    // Guard body
    const gx = guard.col * tileW + tileW / 2;
    const gy = guard.row * tileH + tileH / 2;

    gridCtx.fillStyle = CONFIG.GUARD_BODY;
    gridCtx.beginPath();
    gridCtx.arc(gx, gy, tileW * 0.32, 0, Math.PI * 2);
    gridCtx.fill();
    gridCtx.strokeStyle = 'rgba(0,0,0,0.12)';
    gridCtx.lineWidth = 1;
    gridCtx.stroke();

    // Direction
    const heading = -guard.heading * Math.PI / 180;
    gridCtx.strokeStyle = CONFIG.GUARD_DIR;
    gridCtx.lineWidth = 2;
    gridCtx.lineCap = 'round';
    gridCtx.beginPath();
    gridCtx.moveTo(gx, gy);
    gridCtx.lineTo(gx + Math.cos(heading) * tileW * 0.5,
        gy + Math.sin(heading) * tileH * 0.5);
    gridCtx.stroke();
}

function drawLabel(text, pos, tileW, tileH, color) {
    const x = pos[1] * tileW + tileW / 2;
    const y = pos[0] * tileH + tileH / 2;
    gridCtx.font = `bold ${Math.max(10, tileW * 0.45)}px 'Outfit', sans-serif`;
    gridCtx.textAlign = 'center';
    gridCtx.textBaseline = 'middle';
    // Text shadow for readability on bright tiles
    gridCtx.fillStyle = 'rgba(0,0,0,0.25)';
    gridCtx.fillText(text, x + 1, y + 1);
    gridCtx.fillStyle = color;
    gridCtx.fillText(text, x, y);
}

// ============================================================
// Chart Renderer (Bright theme)
// ============================================================

function renderChart() {
    const ctx = chartCtx;
    const w = chartCanvas.width;
    const h = chartCanvas.height;

    // Clear
    ctx.fillStyle = CONFIG.CHART_BG;
    ctx.fillRect(0, 0, w, h);

    const padding = { top: 22, right: 15, bottom: 25, left: 48 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    let datasets;
    if (state.activeChart === 'rewards') {
        datasets = [
            { data: state.metrics.architect_reward || [], color: '#8b5cf6', label: 'Architect' },
            { data: state.metrics.solver_reward || [], color: '#10b981', label: 'Solver' },
        ];
    } else {
        datasets = [
            { data: state.metrics.solve_rate || [], color: '#10b981', label: 'Solve Rate' },
            { data: state.metrics.detection_rate || [], color: '#f43f5e', label: 'Detect Rate' },
        ];
    }

    // Find Y range
    let yMin = Infinity, yMax = -Infinity;
    datasets.forEach(ds => {
        ds.data.forEach(v => {
            if (v < yMin) yMin = v;
            if (v > yMax) yMax = v;
        });
    });
    if (yMin === Infinity) { yMin = 0; yMax = 1; }
    const yRange = yMax - yMin || 1;
    yMin -= yRange * 0.1;
    yMax += yRange * 0.1;

    const maxLen = Math.max(...datasets.map(d => d.data.length), 1);

    // Grid lines
    ctx.strokeStyle = CONFIG.CHART_GRID;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH * i / 4);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();

        // Y labels
        const val = yMax - (yMax - yMin) * i / 4;
        ctx.fillStyle = CONFIG.CHART_LABEL;
        ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(1), padding.left - 6, y + 4);
    }

    // Draw datasets
    datasets.forEach(ds => {
        if (ds.data.length < 2) return;

        const smoothed = smoothData(ds.data, 5);

        // Line
        ctx.strokeStyle = ds.color;
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        smoothed.forEach((v, i) => {
            const x = padding.left + (i / (maxLen - 1)) * chartW;
            const y = padding.top + (1 - (v - yMin) / (yMax - yMin)) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Area fill — very subtle
        const areaGrad = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
        areaGrad.addColorStop(0, ds.color + '18');  // 10% opacity at top
        areaGrad.addColorStop(1, ds.color + '02');  // ~1% at bottom
        ctx.fillStyle = areaGrad;
        ctx.beginPath();
        smoothed.forEach((v, i) => {
            const x = padding.left + (i / (maxLen - 1)) * chartW;
            const y = padding.top + (1 - (v - yMin) / (yMax - yMin)) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.lineTo(padding.left + chartW, padding.top + chartH);
        ctx.lineTo(padding.left, padding.top + chartH);
        ctx.closePath();
        ctx.fill();
    });

    // Legend
    let legendX = padding.left + 10;
    datasets.forEach(ds => {
        ctx.fillStyle = ds.color;
        ctx.beginPath();
        ctx.roundRect(legendX, padding.top + 3, 14, 4, 2);
        ctx.fill();
        ctx.fillStyle = CONFIG.CHART_LEGEND;
        ctx.font = '11px Outfit';
        ctx.textAlign = 'left';
        ctx.fillText(ds.label, legendX + 18, padding.top + 8);
        legendX += ctx.measureText(ds.label).width + 36;
    });
}

function smoothData(data, window) {
    if (data.length <= window) return data;
    const result = [];
    for (let i = 0; i < data.length; i++) {
        let sum = 0;
        let count = 0;
        for (let j = Math.max(0, i - window); j <= i; j++) {
            sum += data[j];
            count++;
        }
        result.push(sum / count);
    }
    return result;
}

// ============================================================
// UI Updates
// ============================================================

function updateMetrics(metrics) {
    if (!metrics) return;

    setText('metricEpisode', metrics.episode || '—');
    setText('metricSolveRate', ((metrics.solve_rate || 0) * 100).toFixed(0) + '%');
    setText('metricDetectionRate', ((metrics.detection_rate || 0) * 100).toFixed(0) + '%');
    setText('metricArchReward', (metrics.architect_reward || 0).toFixed(2));
    setText('metricSolverReward', (metrics.solver_reward || 0).toFixed(2));
    setText('metricAvgSteps', Math.round(metrics.avg_steps || 0));
    setText('metricBudget', metrics.budget || '—');
}

function updateLayoutStats(envState) {
    if (!envState) return;

    const cameras = envState.cameras ? envState.cameras.length : 0;
    const guards = envState.guards ? envState.guards.length : 0;

    let walls = 0;
    if (envState.grid) {
        envState.grid.forEach(row => row.forEach(t => { if (t === 1) walls++; }));
        const borderWalls = 2 * envState.grid.length + 2 * (envState.grid[0].length - 2);
        walls = Math.max(0, walls - borderWalls);
    }

    setText('statWalls', walls);
    setText('statCameras', cameras);
    setText('statGuards', guards);
    setText('statValid', '✓');
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function setStatus(text, active = false) {
    const indicator = document.getElementById('statusIndicator');
    const statusText = indicator.querySelector('.status-text');
    statusText.textContent = text;
    indicator.classList.toggle('active', active);
}

function showOverlay(show) {
    const overlay = document.getElementById('canvasOverlay');
    overlay.classList.toggle('hidden', !show);
}

// ============================================================
// WebSocket Connection
// ============================================================

let socket;

function initSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('[WS] Connected');
        setStatus('Connected');
    });

    socket.on('disconnect', () => {
        console.log('[WS] Disconnected');
        setStatus('Disconnected');
    });

    socket.on('env_state', (data) => {
        state.envState = data;
        showOverlay(false);
        renderGrid(data);
        updateLayoutStats(data);
        setText('tickCounter', `Tick: ${data.tick || 0}`);
    });

    socket.on('training_update', (data) => {
        const { episode, metrics } = data;

        Object.keys(metrics).forEach(key => {
            if (!state.metrics[key]) state.metrics[key] = [];
            state.metrics[key].push(metrics[key]);
        });

        updateMetrics({ ...metrics, episode });
        renderChart();

        const pct = (episode / state.totalEpisodes * 100).toFixed(0);
        document.getElementById('progressFill').style.width = pct + '%';
        document.getElementById('progressText').textContent = pct + '%';
    });

    socket.on('training_started', (data) => {
        state.isTraining = true;
        state.totalEpisodes = data.episodes;
        setStatus('Training', true);
        document.getElementById('btnTrain').disabled = true;
        document.getElementById('progressContainer').style.display = 'flex';
    });

    socket.on('training_complete', () => {
        state.isTraining = false;
        setStatus('Training Complete');
        document.getElementById('btnTrain').disabled = false;
    });

    socket.on('demo_frame', (data) => {
        state.envState = data.frame;
        showOverlay(false);
        renderGrid(data.frame);
        updateLayoutStats(data.frame);
        setText('tickCounter', `Step: ${data.step}/${data.total}`);
    });

    socket.on('demo_complete', (data) => {
        setStatus(`Demo: ${data.outcome}`);
    });

    socket.on('error', (data) => {
        console.error('[Server Error]', data.message);
        setStatus('Error: ' + data.message);
    });
}

// ============================================================
// Event Listeners
// ============================================================

document.getElementById('btnTrain').addEventListener('click', () => {
    const episodes = parseInt(document.getElementById('inputEpisodes').value) || 500;
    const solverAttempts = parseInt(document.getElementById('inputSolverAttempts').value) || 20;

    state.metrics = {
        episode: [], solve_rate: [], detection_rate: [],
        architect_reward: [], solver_reward: []
    };
    state.totalEpisodes = episodes;

    socket.emit('start_training', { episodes, solver_attempts: solverAttempts });
});

document.getElementById('btnDemo').addEventListener('click', () => {
    setStatus('Running Demo...', true);
    socket.emit('run_demo', {});
});

document.getElementById('toggleVisibility').addEventListener('click', () => {
    state.showVisibility = !state.showVisibility;
    if (state.envState) renderGrid(state.envState);
});

// Chart tab switching
document.querySelectorAll('.chart-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        state.activeChart = tab.dataset.chart;
        renderChart();
    });
});

// ============================================================
// Initialization
// ============================================================

function init() {
    showOverlay(true);
    renderChart();
    initSocket();

    function animate() {
        state.animFrame++;
        requestAnimationFrame(animate);
    }
    animate();
}

init();
