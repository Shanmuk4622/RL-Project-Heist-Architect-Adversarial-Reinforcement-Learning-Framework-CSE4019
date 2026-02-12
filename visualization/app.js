/**
 * Heist Architect — Dashboard Client
 * ====================================
 * Handles WebSocket communication, grid rendering, charts,
 * interactive episodes, and game log display.
 */

// ==================================================================
// Configuration
// ==================================================================

const CONFIG = {
    TILE_COLORS: {
        0: '#1a1f35',   // Empty  (dark navy)
        1: '#4a5568',   // Wall   (slate gray)
        2: '#34d399',   // Start  (neon emerald)
        3: '#fb7185',   // Vault  (hot pink)
        4: '#a78bfa',   // Camera (bright violet)
        5: '#fb923c',   // Guard  (vivid orange)
    },
    SOLVER_COLOR: '#fbbf24',           // Bright gold
    SOLVER_GLOW: 'rgba(251, 191, 36, 0.50)',
    GRID_LINE: 'rgba(99, 130, 255, 0.12)',
    VISION_CONE: 'rgba(232, 121, 249, 0.18)',
    VISION_EDGE: 'rgba(232, 121, 249, 0.4)',
    TRAIL_COLOR: 'rgba(56, 189, 248, 0.45)',
    CHART: {
        BG: '#111528',
        GRID: 'rgba(99, 130, 255, 0.08)',
        LABEL: '#6b75a0',
        SOLVE: '#34d399',
        DETECT: '#fb7185',
        ARCH_REWARD: '#a78bfa',
        SOLVER_REWARD: '#fb923c',
    }
};

// Wall shading constants
const WALL_TOP = '#5a647a';
const WALL_SHADOW = '#2d3348';

// ==================================================================
// State
// ==================================================================

let socket = null;
let gridData = null;
let chartData = { episodes: [], solve_rates: [], detection_rates: [], arch_rewards: [], solver_rewards: [] };
let activeChart = 'rewards';
let showVision = true;
let totalTrainingEpisodes = 500;
let gameLog = [];

// ==================================================================
// DOM Elements
// ==================================================================

const gridCanvas = document.getElementById('gridCanvas');
const gridCtx = gridCanvas.getContext('2d');
const chartCanvas = document.getElementById('chartCanvas');
const chartCtx = chartCanvas.getContext('2d');

const canvasOverlay = document.getElementById('canvasOverlay');
const tickCounter = document.getElementById('tickCounter');
const statusIndicator = document.getElementById('statusIndicator');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const globalEpisodeEl = document.getElementById('globalEpisode');

// Interactive controls
const tempSlider = document.getElementById('intTemperature');
const tempValue = document.getElementById('tempValue');

// ==================================================================
// Initialization
// ==================================================================

function init() {
    socket = io();
    setupSocketEvents();
    setupUIEvents();
    drawEmptyGrid();
    renderChart();

    // Load game log on connect
    fetch('/api/game_log')
        .then(r => r.json())
        .then(data => {
            if (Array.isArray(data) && data.length > 0) {
                gameLog = data;
                renderGameLog();
            }
        })
        .catch(() => { });

    // Load checkpoints
    socket.emit('get_checkpoints');
}

// ==================================================================
// Socket Events
// ==================================================================

function setupSocketEvents() {
    socket.on('connect', () => {
        setStatus('Connected', 'connected');
    });

    socket.on('disconnect', () => {
        setStatus('Disconnected', 'disconnected');
    });

    socket.on('training_started', (data) => {
        totalTrainingEpisodes = data.episodes;
        setStatus('Training...', 'training');
        progressContainer.style.display = 'flex';
        canvasOverlay.style.display = 'none';
        setButtonsDisabled(true);

        if (data.resume_from > 0) {
            console.log(`Resuming from episode ${data.resume_from}`);
        }
    });

    socket.on('training_update', (data) => {
        updateMetrics(data.metrics, data.episode);
        updateProgress(data.episode);

        // Update chart data
        chartData.episodes.push(data.episode);
        chartData.solve_rates.push(data.metrics.solve_rate);
        chartData.detection_rates.push(data.metrics.detection_rate);
        chartData.arch_rewards.push(data.metrics.architect_reward);
        chartData.solver_rewards.push(data.metrics.solver_reward);
        renderChart();
    });

    socket.on('training_complete', (data) => {
        setStatus('Training Complete', 'complete');
        progressContainer.style.display = 'none';
        setButtonsDisabled(false);

        if (data.global_episode) {
            globalEpisodeEl.textContent = data.global_episode;
        }
        if (data.game_log) {
            // Merge latest entries
            gameLog = gameLog.concat(data.game_log.filter(e =>
                !gameLog.some(existing => existing.episode === e.episode)
            ));
            renderGameLog();
        }
    });

    socket.on('env_state', (data) => {
        gridData = data;
        renderGrid(data);
        if (data.tick !== undefined) {
            tickCounter.textContent = `Tick: ${data.tick}`;
        }
    });

    socket.on('demo_started', () => {
        setStatus('Demo Running...', 'training');
        canvasOverlay.style.display = 'none';
    });

    socket.on('demo_frame', (data) => {
        renderGrid(data.frame);
        tickCounter.textContent = `Step: ${data.step + 1}/${data.total}`;
    });

    socket.on('demo_complete', (data) => {
        setStatus(`Demo: ${data.outcome}`, 'complete');
        setButtonsDisabled(false);
    });

    socket.on('interactive_started', (data) => {
        setStatus(`Interactive: ${data.num_episodes} ep (budget=${data.budget})`, 'training');
        canvasOverlay.style.display = 'none';
        setButtonsDisabled(true);
    });

    socket.on('interactive_complete', (data) => {
        setStatus('Interactive Complete', 'complete');
        setButtonsDisabled(false);
        if (data.global_episode) {
            globalEpisodeEl.textContent = data.global_episode;
        }
    });

    socket.on('game_log_entry', (entry) => {
        // Add single entry
        if (!gameLog.some(e => e.episode === entry.episode)) {
            gameLog.push(entry);
            renderGameLog();
        }
    });

    socket.on('game_log_update', (data) => {
        if (data.log && data.log.length > 0) {
            gameLog = data.log;
            renderGameLog();
        }
        if (data.global_episode) {
            globalEpisodeEl.textContent = data.global_episode;
        }
    });

    socket.on('error', (data) => {
        setStatus(`Error: ${data.message}`, 'error');
        setButtonsDisabled(false);
    });

    socket.on('checkpoints_list', (data) => {
        const select = document.getElementById('simCheckpoint');

        if (!data.checkpoints || data.checkpoints.length === 0) {
            select.innerHTML = '<option value="" disabled selected>No checkpoints found</option>';
            return;
        }

        select.innerHTML = '<option value="" disabled>Select Episode...</option>';
        data.checkpoints.forEach(ep => {
            const opt = document.createElement('option');
            opt.value = ep;
            opt.text = `Episode ${ep}`;
            select.add(opt);
        });

        // Auto-select latest
        select.value = data.checkpoints[data.checkpoints.length - 1];
    });

    socket.on('simulation_started', (data) => {
        setStatus(`Simulating Ep ${data.episode} (Budget ${data.budget})...`, 'training');
        canvasOverlay.style.display = 'none';
        setButtonsDisabled(true);
    });

    socket.on('simulation_frame', (data) => {
        renderGrid(data.frame);
        tickCounter.textContent = `Step: ${data.step + 1}/${data.total}`;
    });

    socket.on('simulation_complete', (data) => {
        const outcomeEmoji = data.outcome === 'vault_reached' ? '🏆' : (data.outcome === 'detected' ? '🚨' : '⏱️');
        setStatus(`Simulation: ${outcomeEmoji} ${data.outcome} (Reward: ${data.reward.toFixed(2)})`, 'complete');
        setButtonsDisabled(false);
    });
}

// ==================================================================
// UI Events
// ==================================================================

function setupUIEvents() {
    // Training button
    document.getElementById('btnTrain').addEventListener('click', () => {
        const episodes = parseInt(document.getElementById('inputEpisodes').value) || 500;
        const solverAttempts = parseInt(document.getElementById('inputSolverAttempts').value) || 20;

        socket.emit('start_training', {
            episodes: episodes,
            solver_attempts: solverAttempts,
            resume: true,
        });
    });

    // Demo button
    document.getElementById('btnDemo').addEventListener('click', () => {
        setButtonsDisabled(true);
        socket.emit('run_demo');
    });

    // Interactive button
    document.getElementById('btnInteractive').addEventListener('click', () => {
        const params = {
            num_episodes: parseInt(document.getElementById('intEpisodes').value) || 1,
            budget: parseInt(document.getElementById('intBudget').value) || 15,
            freeze_architect: document.getElementById('intFreezeArchitect').checked,
            freeze_solver: document.getElementById('intFreezeSolver').checked,
            temperature: parseFloat(tempSlider.value) || 1.0,
            solver_attempts: parseInt(document.getElementById('intSolverAttempts').value) || 20,
            allow_cameras: document.getElementById('intAllowCameras').checked,
            allow_guards: document.getElementById('intAllowGuards').checked,
        };

        socket.emit('run_interactive', params);
    });

    // Simulate button
    const btnSimulate = document.getElementById('btnSimulate');
    if (btnSimulate) {
        btnSimulate.addEventListener('click', () => {
            const episode = parseInt(document.getElementById('simCheckpoint').value);
            if (!episode) return;

            const budget = parseInt(document.getElementById('simBudget').value) || 15;
            const attempts = parseInt(document.getElementById('simAttempts').value) || 5;

            socket.emit('run_simulation', {
                episode: episode,
                budget: budget,
                solver_attempts: attempts
            });
        });
    }

    // Toggle vision
    document.getElementById('toggleVisibility').addEventListener('click', () => {
        showVision = !showVision;
        if (gridData) renderGrid(gridData);
    });

    // Temperature slider
    tempSlider.addEventListener('input', () => {
        tempValue.textContent = parseFloat(tempSlider.value).toFixed(1);
    });

    // Chart tabs
    document.querySelectorAll('.chart-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            activeChart = tab.dataset.chart;
            renderChart();
        });
    });
}

// ==================================================================
// Grid Rendering
// ==================================================================

function drawEmptyGrid() {
    const size = gridCanvas.width;
    gridCtx.fillStyle = '#0f1225';
    gridCtx.fillRect(0, 0, size, size);

    const cells = 20;
    const cellSize = size / cells;

    gridCtx.strokeStyle = CONFIG.GRID_LINE;
    gridCtx.lineWidth = 0.5;

    for (let i = 0; i <= cells; i++) {
        gridCtx.beginPath();
        gridCtx.moveTo(i * cellSize, 0);
        gridCtx.lineTo(i * cellSize, size);
        gridCtx.stroke();

        gridCtx.beginPath();
        gridCtx.moveTo(0, i * cellSize);
        gridCtx.lineTo(size, i * cellSize);
        gridCtx.stroke();
    }
}

function renderGrid(data) {
    if (!data || !data.grid) return;

    const grid = data.grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const cellW = gridCanvas.width / cols;
    const cellH = gridCanvas.height / rows;

    // Clear
    gridCtx.fillStyle = '#0f1225';
    gridCtx.fillRect(0, 0, gridCanvas.width, gridCanvas.height);

    // Draw tiles
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = grid[r][c];
            const x = c * cellW;
            const y = r * cellH;

            gridCtx.fillStyle = CONFIG.TILE_COLORS[val] || '#e8ecf4';
            gridCtx.fillRect(x, y, cellW, cellH);

            // Wall 3D effect
            if (val === 1) {
                gridCtx.fillStyle = WALL_TOP;
                gridCtx.fillRect(x, y, cellW, cellH * 0.3);
                gridCtx.fillStyle = WALL_SHADOW;
                gridCtx.fillRect(x, y + cellH * 0.85, cellW, cellH * 0.15);
            }

            // Camera indicator
            if (val === 4) {
                gridCtx.fillStyle = 'rgba(139, 92, 246, 0.35)';
                gridCtx.beginPath();
                gridCtx.arc(x + cellW / 2, y + cellH / 2, cellW * 0.25, 0, Math.PI * 2);
                gridCtx.fill();
            }
        }
    }

    // Draw visibility/vision cones (visibility is a 2D array from environment)
    if (showVision && data.visibility) {
        gridCtx.fillStyle = CONFIG.VISION_CONE;
        for (let r = 0; r < data.visibility.length; r++) {
            for (let c = 0; c < data.visibility[r].length; c++) {
                if (data.visibility[r][c] > 0) {
                    gridCtx.fillRect(c * cellW, r * cellH, cellW, cellH);
                }
            }
        }
    }

    // Draw solver path (the trail from start to current position)
    if (data.solver_path && data.solver_path.length > 1) {
        // Draw connecting line
        gridCtx.strokeStyle = '#22d3ee';
        gridCtx.lineWidth = 2.5;
        gridCtx.lineJoin = 'round';
        gridCtx.lineCap = 'round';
        gridCtx.setLineDash([]);
        gridCtx.beginPath();
        const [startR, startC] = data.solver_path[0];
        gridCtx.moveTo(startC * cellW + cellW / 2, startR * cellH + cellH / 2);
        for (let i = 1; i < data.solver_path.length; i++) {
            const [pr, pc] = data.solver_path[i];
            gridCtx.lineTo(pc * cellW + cellW / 2, pr * cellH + cellH / 2);
        }
        gridCtx.stroke();

        // Draw dots at each step
        data.solver_path.forEach(([r, c], idx) => {
            const alpha = 0.3 + (idx / data.solver_path.length) * 0.7;
            gridCtx.fillStyle = `rgba(34, 211, 238, ${alpha})`;
            gridCtx.beginPath();
            gridCtx.arc(c * cellW + cellW / 2, r * cellH + cellH / 2, cellW * 0.15, 0, Math.PI * 2);
            gridCtx.fill();
        });
    }

    // Draw solver
    if (data.solver_pos) {
        const [sr, sc] = data.solver_pos;
        const sx = sc * cellW;
        const sy = sr * cellH;

        // Glow
        gridCtx.fillStyle = CONFIG.SOLVER_GLOW;
        gridCtx.beginPath();
        gridCtx.arc(sx + cellW / 2, sy + cellH / 2, cellW * 0.7, 0, Math.PI * 2);
        gridCtx.fill();

        // Solver
        gridCtx.fillStyle = CONFIG.SOLVER_COLOR;
        gridCtx.beginPath();
        gridCtx.arc(sx + cellW / 2, sy + cellH / 2, cellW * 0.35, 0, Math.PI * 2);
        gridCtx.fill();

        // Border
        gridCtx.strokeStyle = 'rgba(161, 98, 7, 0.5)';
        gridCtx.lineWidth = 1.5;
        gridCtx.stroke();
    }

    // Grid lines
    gridCtx.strokeStyle = CONFIG.GRID_LINE;
    gridCtx.lineWidth = 0.5;
    for (let r = 0; r <= rows; r++) {
        gridCtx.beginPath();
        gridCtx.moveTo(0, r * cellH);
        gridCtx.lineTo(gridCanvas.width, r * cellH);
        gridCtx.stroke();
    }
    for (let c = 0; c <= cols; c++) {
        gridCtx.beginPath();
        gridCtx.moveTo(c * cellW, 0);
        gridCtx.lineTo(c * cellW, gridCanvas.height);
        gridCtx.stroke();
    }
}

// ==================================================================
// Chart Rendering
// ==================================================================

function renderChart() {
    const W = chartCanvas.width;
    const H = chartCanvas.height;
    const pad = { top: 15, right: 10, bottom: 25, left: 45 };

    chartCtx.fillStyle = CONFIG.CHART.BG;
    chartCtx.fillRect(0, 0, W, H);

    let series1, series2, color1, color2, label1, label2;

    if (activeChart === 'rewards') {
        series1 = chartData.arch_rewards;
        series2 = chartData.solver_rewards;
        color1 = CONFIG.CHART.ARCH_REWARD;
        color2 = CONFIG.CHART.SOLVER_REWARD;
        label1 = 'Architect';
        label2 = 'Solver';
    } else {
        series1 = chartData.solve_rates;
        series2 = chartData.detection_rates;
        color1 = CONFIG.CHART.SOLVE;
        color2 = CONFIG.CHART.DETECT;
        label1 = 'Solve Rate';
        label2 = 'Detect Rate';
    }

    if (series1.length === 0) {
        chartCtx.fillStyle = CONFIG.CHART.LABEL;
        chartCtx.font = '12px Outfit';
        chartCtx.textAlign = 'center';
        chartCtx.fillText('No data yet', W / 2, H / 2);
        return;
    }

    const allVals = [...series1, ...series2];
    const minVal = Math.min(0, ...allVals);
    const maxVal = Math.max(1, ...allVals) * 1.05;
    const range = maxVal - minVal || 1;

    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Grid lines
    chartCtx.strokeStyle = CONFIG.CHART.GRID;
    chartCtx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH / 4) * i;
        chartCtx.beginPath();
        chartCtx.moveTo(pad.left, y);
        chartCtx.lineTo(W - pad.right, y);
        chartCtx.stroke();

        const val = maxVal - (range / 4) * i;
        chartCtx.fillStyle = CONFIG.CHART.LABEL;
        chartCtx.font = '10px JetBrains Mono';
        chartCtx.textAlign = 'right';
        chartCtx.fillText(val.toFixed(1), pad.left - 5, y + 3);
    }

    function drawLine(series, color) {
        if (series.length < 2) return;

        // Gradient fill
        const grad = chartCtx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
        grad.addColorStop(0, color + '18');
        grad.addColorStop(1, color + '02');

        chartCtx.beginPath();
        chartCtx.moveTo(pad.left, pad.top + plotH);

        for (let i = 0; i < series.length; i++) {
            const x = pad.left + (i / Math.max(series.length - 1, 1)) * plotW;
            const y = pad.top + plotH - ((series[i] - minVal) / range) * plotH;
            chartCtx.lineTo(x, y);
        }

        chartCtx.lineTo(pad.left + plotW, pad.top + plotH);
        chartCtx.closePath();
        chartCtx.fillStyle = grad;
        chartCtx.fill();

        // Line
        chartCtx.beginPath();
        chartCtx.strokeStyle = color;
        chartCtx.lineWidth = 2;

        for (let i = 0; i < series.length; i++) {
            const x = pad.left + (i / Math.max(series.length - 1, 1)) * plotW;
            const y = pad.top + plotH - ((series[i] - minVal) / range) * plotH;
            i === 0 ? chartCtx.moveTo(x, y) : chartCtx.lineTo(x, y);
        }
        chartCtx.stroke();
    }

    drawLine(series1, color1);
    drawLine(series2, color2);

    // Legend
    const legendY = H - 6;
    chartCtx.font = '10px Outfit';
    chartCtx.textAlign = 'left';

    chartCtx.fillStyle = color1;
    chartCtx.fillRect(pad.left, legendY - 5, 12, 3);
    chartCtx.fillStyle = CONFIG.CHART.LABEL;
    chartCtx.fillText(label1, pad.left + 16, legendY);

    chartCtx.fillStyle = color2;
    chartCtx.fillRect(pad.left + 90, legendY - 5, 12, 3);
    chartCtx.fillStyle = CONFIG.CHART.LABEL;
    chartCtx.fillText(label2, pad.left + 106, legendY);
}

// ==================================================================
// Game Log Rendering
// ==================================================================

function renderGameLog() {
    const tbody = document.getElementById('gameLogBody');
    const logCount = document.getElementById('logCount');

    logCount.textContent = `${gameLog.length} episodes`;

    if (gameLog.length === 0) {
        tbody.innerHTML = '<tr class="log-empty"><td colspan="10">No episodes yet — start training or run interactive episodes</td></tr>';
        return;
    }

    // Show last 100 entries (most recent first)
    const entries = gameLog.slice(-100).reverse();

    tbody.innerHTML = entries.map(e => {
        const solveClass = e.solve_rate >= 0.5 ? 'val-good' : (e.solve_rate > 0 ? 'val-neutral' : 'val-bad');
        const detectClass = e.detection_rate >= 0.5 ? 'val-bad' : (e.detection_rate > 0 ? 'val-neutral' : 'val-good');

        let modeLabel, modeClass;
        if (e.is_interactive) {
            if (e.freeze_architect || e.freeze_solver) {
                modeLabel = e.freeze_architect ? '❄️A' : '❄️S';
                modeClass = 'frozen';
            } else {
                modeLabel = '🎮';
                modeClass = 'interactive';
            }
        } else {
            modeLabel = 'Auto';
            modeClass = 'auto';
        }

        const layoutStr = `${e.walls}W ${e.cameras}C ${e.guards}G`;

        return `<tr>
            <td>${e.episode}</td>
            <td>${e.phase || '—'}</td>
            <td>${e.budget}</td>
            <td>${layoutStr}</td>
            <td class="${solveClass}">${(e.solve_rate * 100).toFixed(0)}%</td>
            <td class="${detectClass}">${(e.detection_rate * 100).toFixed(0)}%</td>
            <td>${e.architect_reward >= 0 ? '+' : ''}${e.architect_reward.toFixed(2)}</td>
            <td>${e.solver_reward >= 0 ? '+' : ''}${e.solver_reward.toFixed(2)}</td>
            <td>${e.avg_steps.toFixed(0)}</td>
            <td><span class="log-mode ${modeClass}">${modeLabel}</span></td>
        </tr>`;
    }).join('');

    // Auto-scroll to top (newest)
    document.getElementById('gameLogContainer').scrollTop = 0;
}

// ==================================================================
// UI Helpers
// ==================================================================

function setStatus(text, state) {
    const dot = statusIndicator.querySelector('.status-dot');
    const textEl = statusIndicator.querySelector('.status-text');
    textEl.textContent = text;

    statusIndicator.className = `status-indicator ${state}`;
}

function updateMetrics(metrics, episode) {
    document.getElementById('metricEpisode').textContent = episode;
    document.getElementById('metricSolveRate').textContent = (metrics.solve_rate * 100).toFixed(0) + '%';
    document.getElementById('metricDetectionRate').textContent = (metrics.detection_rate * 100).toFixed(0) + '%';
    document.getElementById('metricArchReward').textContent = metrics.architect_reward.toFixed(2);
    document.getElementById('metricSolverReward').textContent = metrics.solver_reward.toFixed(2);
    document.getElementById('metricAvgSteps').textContent = metrics.avg_steps.toFixed(0);
    document.getElementById('metricBudget').textContent = metrics.budget;
    globalEpisodeEl.textContent = episode;
}

function updateProgress(episode) {
    const pct = Math.min(100, (episode / totalTrainingEpisodes) * 100);
    progressFill.style.width = pct + '%';
    progressText.textContent = `${pct.toFixed(0)}% (${episode}/${totalTrainingEpisodes})`;
}

function setButtonsDisabled(disabled) {
    document.getElementById('btnTrain').disabled = disabled;
    document.getElementById('btnDemo').disabled = disabled;
    document.getElementById('btnInteractive').disabled = disabled;
    const btnSim = document.getElementById('btnSimulate');
    if (btnSim) btnSim.disabled = disabled;
}

// ==================================================================
// Bootstrap
// ==================================================================

document.addEventListener('DOMContentLoaded', init);
