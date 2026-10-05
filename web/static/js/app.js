const KALSHI_COLOR = "#5eb5ff";
const POLY_COLOR = "#ffb020";
const BASIS_COLOR = "#c084fc";

const CHART_LAYOUT = {
  layout: { background: { color: "#0c1014" }, textColor: "#d7dce2", fontFamily: "Consolas, Monaco, monospace" },
  grid: { vertLines: { color: "rgba(255,255,255,0.06)" }, horzLines: { color: "rgba(255,255,255,0.06)" } },
  rightPriceScale: { borderColor: "#2a3238" },
  timeScale: { borderColor: "#2a3238", timeVisible: true, secondsVisible: false },
  crosshair: { vertLine: { color: "#556", style: 3, width: 1 }, horzLine: { color: "#556", style: 3, width: 1 } },
};

let board = null;
let selected = null;
let priceChart = null;
let basisChart = null;
let kSeries = null;
let pSeries = null;
let bSeries = null;
let boardTimer = null;
let seriesTimer = null;

function $(id) {
  return document.getElementById(id);
}

function fmtPct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function fmtEdge(value) {
  if (value == null || Number.isNaN(value)) return "—";
  const bps = Number(value) * 100;
  return `${bps >= 0 ? "+" : ""}${bps.toFixed(1)}¢`;
}

function yesOf(side) {
  if (!side) return null;
  return side.yes_mid ?? side.yes_ask ?? side.yes_bid ?? null;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function tickClock() {
  $("clock").textContent = new Date().toLocaleTimeString();
}

function midFormat() {
  return { type: "custom", minMove: 0.0001, formatter: (price) => `${(Number(price) * 100).toFixed(1)}%` };
}

function ensureCharts() {
  if (priceChart) return;
  const priceEl = $("price-chart");
  priceChart = LightweightCharts.createChart(priceEl, { ...CHART_LAYOUT, width: priceEl.clientWidth, height: 280 });
  kSeries = priceChart.addLineSeries({ color: KALSHI_COLOR, lineWidth: 2, priceFormat: midFormat() });
  pSeries = priceChart.addLineSeries({ color: POLY_COLOR, lineWidth: 2, priceFormat: midFormat() });
  const basisEl = $("basis-chart");
  basisChart = LightweightCharts.createChart(basisEl, { ...CHART_LAYOUT, width: basisEl.clientWidth, height: 140 });
  bSeries = basisChart.addLineSeries({ color: BASIS_COLOR, lineWidth: 2, priceFormat: midFormat() });
}

function currentChoice() {
  const picked = document.querySelector('input[name="choice"]:checked');
  return picked && picked.value === "no" ? "no" : "yes";
}

function renderContracts() {
  const q = ($("contract-filter").value || "").toLowerCase();
  const list = $("contract-list");
  list.innerHTML = "";
  (board?.markets || []).forEach((row) => {
    if (q && !row.label.toLowerCase().includes(q) && !row.id.includes(q)) return;
    const el = document.createElement("button");
    el.type = "button";
    el.className = `contract${selected === row.id ? " active" : ""}`;
    const k = yesOf(row.kalshi);
    const p = yesOf(row.polymarket);
    const edge = row.net_edge;
    el.innerHTML = `<span class="name">${row.label}</span><span class="px k">${fmtPct(k)}</span><span class="px p">${fmtPct(p)}</span>`;
    if (row.tradable) el.classList.add("hot");
    el.addEventListener("click", () => selectMarket(row.id));
    list.appendChild(el);
  });
}

function renderArbTable() {
  const body = $("arb-body");
  body.innerHTML = "";
  (board?.markets || []).forEach((row) => {
    const tr = document.createElement("tr");
    if (row.tradable) tr.classList.add("hot");
    if (row.id === selected) tr.classList.add("active");
    const best = row.best || {};
    tr.innerHTML = `
      <td>${row.label}</td>
      <td>${fmtPct(yesOf(row.kalshi))}</td>
      <td>${fmtPct(yesOf(row.polymarket))}</td>
      <td>${row.basis == null ? "—" : (row.basis >= 0 ? "+" : "") + (row.basis * 100).toFixed(1) + "¢"}</td>
      <td>${best.label || "—"}</td>
      <td>${fmtEdge(row.net_edge)}</td>
      <td>${row.quality || "—"}</td>`;
    tr.addEventListener("click", () => selectMarket(row.id));
    body.appendChild(tr);
  });
}

function paintInspector(row, series) {
  $("sel-name").textContent = row?.label || "—";
  $("sel-kalshi").textContent = fmtPct(yesOf(row?.kalshi));
  $("sel-poly").textContent = fmtPct(yesOf(row?.polymarket));
  $("sel-basis").textContent = row?.basis == null ? "—" : `${row.basis >= 0 ? "+" : ""}${(row.basis * 100).toFixed(1)}¢`;
  $("sel-edge").textContent = fmtEdge(row?.net_edge);
  $("sel-combo").textContent = row?.best?.label || "No complementary combo";
  $("sel-kvol").textContent = series?.kalshi_volume_label || "—";
  $("sel-pvol").textContent = series?.polymarket_volume_label || "—";
}

function resetModel() {
  $("ml-direction").textContent = "—";
  $("ml-direction").className = "backend-output";
  $("ml-prob").textContent = "—";
  $("ml-leader").textContent = "—";
  $("ml-acc").textContent = "—";
  $("ml-source").textContent = "—";
}

async function loadEvents() {
  const data = await fetchJson("/api/events");
  const select = $("event-select");
  select.innerHTML = "";
  data.events.forEach((event) => {
    const option = document.createElement("option");
    option.value = event.id;
    option.textContent = event.label;
    if (event.id === data.default_event) option.selected = true;
    select.appendChild(option);
  });
}

async function loadBoard() {
  const eventId = $("event-select").value;
  board = await fetchJson(`/api/board?event_id=${encodeURIComponent(eventId)}`);
  $("tradable-count").textContent = String(board.tradable_count ?? 0);
  $("market-count").textContent = String(board.market_count ?? 0);
  if (!selected && board.markets?.length) {
    const hot = board.markets.find((m) => m.tradable) || board.markets.find((m) => m.focus) || board.markets[0];
    selected = hot.id;
  }
  renderContracts();
  renderArbTable();
  const row = (board.markets || []).find((m) => m.id === selected);
  paintInspector(row);
  $("status-line").textContent = `Live API · ${new Date().toLocaleTimeString()} · ${board.market_count} paired contracts`;
}

function toPoints(points) {
  return (points || []).map((pt) => ({ time: pt.time, value: pt.value }));
}

async function loadSeries() {
  if (!selected) return;
  const eventId = $("event-select").value;
  const series = await fetchJson(
    `/api/series?event_id=${encodeURIComponent(eventId)}&market=${encodeURIComponent(selected)}&choice=${currentChoice()}`
  );
  ensureCharts();
  $("price-title").textContent = `${series.choice.toUpperCase()} · ${series.market_label}`;
  kSeries.setData(toPoints(series.kalshi));
  pSeries.setData(toPoints(series.polymarket));
  const basis = [];
  const kMap = new Map((series.kalshi || []).map((pt) => [pt.time, pt.value]));
  (series.polymarket || []).forEach((pt) => {
    if (kMap.has(pt.time)) basis.push({ time: pt.time, value: kMap.get(pt.time) - pt.value });
  });
  bSeries.setData(basis);
  priceChart.timeScale().fitContent();
  basisChart.timeScale().fitContent();
  $("price-empty").classList.toggle("hidden", series.kalshi.length + series.polymarket.length > 0);
  $("price-legend").textContent = `Kalshi ${fmtPct(series.kalshi.at(-1)?.value)}  ·  Poly ${fmtPct(series.polymarket.at(-1)?.value)}  ·  ${series.kalshi_points}/${series.polymarket_points} pts`;
  $("basis-legend").textContent = basis.length ? `Last ${(basis.at(-1).value * 100).toFixed(1)}¢` : "";
  const row = (board?.markets || []).find((m) => m.id === selected);
  paintInspector(row, series);
  await loadPredict();
}

async function loadPredict() {
  if (!selected) return;
  try {
    const eventId = $("event-select").value;
    const data = await fetchJson(
      `/api/predict?event_id=${encodeURIComponent(eventId)}&market=${encodeURIComponent(selected)}&threshold=${encodeURIComponent($("threshold").value || "0.08")}`
    );
    const fmt = data.formatted || {};
    $("ml-direction").textContent = fmt.direction || "—";
    $("ml-direction").className = `backend-output ${fmt.css || ""}`;
    $("ml-prob").textContent = fmt.prob_compress || "—";
    $("ml-leader").textContent = fmt.leader || "—";
    $("ml-acc").textContent = fmt.recent_accuracy || "—";
    $("ml-source").textContent = fmt.source || "—";
    if (data.path?.basis?.length && bSeries) {
      bSeries.setData(toPoints(data.path.basis));
      basisChart.timeScale().fitContent();
    }
  } catch (err) {
    resetModel();
    $("ml-direction").textContent = err.message;
  }
}

async function selectMarket(id) {
  selected = id;
  renderContracts();
  renderArbTable();
  await loadSeries();
}

function startTimers() {
  if (boardTimer) clearInterval(boardTimer);
  if (seriesTimer) clearInterval(seriesTimer);
  boardTimer = setInterval(loadBoard, 45000);
  seriesTimer = setInterval(loadSeries, 60000);
}

document.addEventListener("DOMContentLoaded", async () => {
  tickClock();
  setInterval(tickClock, 1000);
  window.addEventListener("resize", () => {
    if (priceChart) priceChart.applyOptions({ width: $("price-chart").clientWidth });
    if (basisChart) basisChart.applyOptions({ width: $("basis-chart").clientWidth });
  });
  $("event-select").addEventListener("change", async () => {
    selected = null;
    await loadBoard();
    await loadSeries();
  });
  $("contract-filter").addEventListener("input", renderContracts);
  document.querySelectorAll('input[name="choice"]').forEach((el) => {
    el.addEventListener("change", loadSeries);
  });
  $("threshold").addEventListener("change", loadPredict);
  try {
    await loadEvents();
    await loadBoard();
    await loadSeries();
    startTimers();
  } catch (err) {
    $("status-line").textContent = err.message;
  }
});
