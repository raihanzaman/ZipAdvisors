const KALSHI_COLOR = "#5eb5ff";
const POLY_COLOR = "#ffb020";
const VOL_COLOR = "#c084fc";
const UP_COLOR = "#3dd68c";
const DOWN_COLOR = "#ff6b6b";

const CHART_LAYOUT = {
  layout: {
    background: { color: "#101418" },
    textColor: "#d7dce2",
    fontFamily: "Consolas, Monaco, monospace",
  },
  grid: {
    vertLines: { color: "rgba(255,255,255,0.06)" },
    horzLines: { color: "rgba(255,255,255,0.06)" },
  },
  rightPriceScale: { borderColor: "#2a3238" },
  timeScale: {
    borderColor: "#2a3238",
    timeVisible: true,
    secondsVisible: false,
  },
  crosshair: {
    vertLine: { color: "#556", style: 3, width: 1 },
    horzLine: { color: "#556", style: 3, width: 1 },
  },
};

const charts = {
  kalshi: null,
  polymarket: null,
  prob: null,
};

let pollTimer = null;

function $(id) {
  return document.getElementById(id);
}

function toPercentPoints(points) {
  return (points || []).map((point) => ({ time: point.time, value: point.value }));
}

function priceFormat() {
  return {
    type: "custom",
    minMove: 0.0001,
    formatter: (price) => `${(Number(price) * 100).toFixed(1)}%`,
  };
}

function volFormat() {
  return {
    type: "custom",
    minMove: 0.0001,
    formatter: (price) => Number(price).toFixed(3),
  };
}

function lastValue(points) {
  if (!points || !points.length) return null;
  return points[points.length - 1].value;
}

function fmtPct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function createChart(containerId) {
  const el = $(containerId);
  const chart = LightweightCharts.createChart(el, {
    ...CHART_LAYOUT,
    width: el.clientWidth,
    height: el.clientHeight,
  });
  const price = chart.addLineSeries({
    color: KALSHI_COLOR,
    lineWidth: 2,
    priceFormat: priceFormat(),
  });
  const overlay = chart.addLineSeries({
    color: POLY_COLOR,
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    priceFormat: priceFormat(),
  });
  return { chart, price, overlay, el };
}

function resizeChart(entry) {
  if (!entry) return;
  entry.chart.applyOptions({ width: entry.el.clientWidth, height: entry.el.clientHeight });
}

function destroyCharts() {
  Object.values(charts).forEach((entry) => {
    if (entry && entry.chart) entry.chart.remove();
  });
  charts.kalshi = null;
  charts.polymarket = null;
  charts.prob = null;
}

function ensureCharts(showProb) {
  if (!charts.kalshi) charts.kalshi = createChart("kalshi-chart");
  if (!charts.polymarket) charts.polymarket = createChart("polymarket-chart");
  $("prob-section").classList.toggle("hidden", !showProb);
  if (showProb && !charts.prob) {
    charts.prob = createChart("prob-chart");
    charts.prob.price.applyOptions({
      color: UP_COLOR,
      priceFormat: priceFormat(),
    });
    charts.prob.overlay.applyOptions({ visible: false });
  }
  resizeChart(charts.kalshi);
  resizeChart(charts.polymarket);
  if (showProb) resizeChart(charts.prob);
}

function setEmpty(side, empty) {
  $(`${side}-empty`).style.display = empty ? "block" : "none";
}

function setLegend(id, text) {
  $(id).textContent = text || "";
}

function resetStats() {
  $("trading-volume").textContent = "N/A";
  $("xgb-direction").textContent = "N/A";
  $("xgb-direction").className = "backend-output";
  $("xgb-prob").textContent = "N/A";
  $("xgb-confidence").textContent = "N/A";
  $("xgb-accuracy").textContent = "N/A";
  $("xgb-basis").textContent = "N/A";
}

function applyPrediction(formatted) {
  if (!formatted) {
    resetStats();
    return;
  }
  $("xgb-direction").textContent = formatted.direction;
  $("xgb-direction").className = `backend-output ${formatted.css || ""}`;
  $("xgb-prob").textContent = formatted.prob_up;
  $("xgb-confidence").textContent = formatted.confidence;
  $("xgb-accuracy").textContent = formatted.recent_accuracy;
  $("xgb-basis").textContent = formatted.basis;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
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
  await loadMarkets();
}

async function loadMarkets() {
  const eventId = $("event-select").value;
  const select = $("market-select");
  select.innerHTML = '<option value="">Select contract</option>';
  if (!eventId) return;
  const data = await fetchJson(`/api/markets?event_id=${encodeURIComponent(eventId)}`);
  data.markets.forEach((market) => {
    const option = document.createElement("option");
    option.value = market.id;
    option.textContent = market.xgb ? `${market.label} · XGB` : market.label;
    select.appendChild(option);
  });
    const preferred = data.markets.find((m) => m.xgb) || data.markets[0];
    if (preferred) select.value = preferred.id;
    select.dispatchEvent(new Event("change"));
}

function currentTarget() {
  const selected = document.querySelector('input[name="market"]:checked');
  return selected && selected.value === "polymarket" ? "polymarket" : "kalshi";
}

function currentChoice() {
  const selected = document.querySelector('input[name="choice"]:checked');
  return selected && selected.value === "no" ? "no" : "yes";
}

function paintPrice(entry, primary, overlay, primaryColor, overlayColor, overlayIsVol) {
  entry.price.applyOptions({
    color: primaryColor,
    priceFormat: priceFormat(),
  });
  entry.overlay.applyOptions({
    color: overlayColor,
    visible: overlay && overlay.length > 0,
    priceFormat: overlayIsVol ? volFormat() : priceFormat(),
  });
  entry.price.setData(toPercentPoints(primary));
  entry.overlay.setData(toPercentPoints(overlay));
  entry.chart.timeScale().fitContent();
}

async function refresh() {
  const eventId = $("event-select").value;
  const market = $("market-select").value;
  const mode = $("algo-select").value;
  const status = $("status-line");
  if (!eventId || !market) {
    status.textContent = "Select an event and contract.";
    return;
  }

  try {
    const series = await fetchJson(
      `/api/series?event_id=${encodeURIComponent(eventId)}&market=${encodeURIComponent(market)}&choice=${currentChoice()}`
    );
    ensureCharts(mode === "xgboost");
    const marketTitle = series.market_label || market;
    $("kalshi-title").textContent = `Kalshi · ${marketTitle}`;
    $("polymarket-title").textContent = `Polymarket · ${marketTitle}`;
    $("trading-volume").textContent = series.volume_label || "N/A";

    if (mode === "volatility") {
      paintPrice(charts.kalshi, series.kalshi, series.kalshi_vol, KALSHI_COLOR, VOL_COLOR, true);
      paintPrice(charts.polymarket, series.polymarket, series.polymarket_vol, POLY_COLOR, VOL_COLOR, true);
      setLegend("kalshi-legend", `YES ${fmtPct(lastValue(series.kalshi))}  ·  rolling vol overlay`);
      setLegend("polymarket-legend", `YES ${fmtPct(lastValue(series.polymarket))}  ·  rolling vol overlay`);
    } else {
      paintPrice(charts.kalshi, series.kalshi, series.polymarket, KALSHI_COLOR, POLY_COLOR, false);
      paintPrice(charts.polymarket, series.polymarket, series.kalshi, POLY_COLOR, KALSHI_COLOR, false);
      setLegend(
        "kalshi-legend",
        `Kalshi ${fmtPct(lastValue(series.kalshi))}  ·  Poly ${fmtPct(lastValue(series.polymarket))}`
      );
      setLegend(
        "polymarket-legend",
        `Poly ${fmtPct(lastValue(series.polymarket))}  ·  Kalshi ${fmtPct(lastValue(series.kalshi))}`
      );
    }

    setEmpty("kalshi", !series.kalshi.length);
    setEmpty("polymarket", !series.polymarket.length);

    if (mode === "xgboost") {
      const predict = await fetchJson(
        `/api/predict?event_id=${encodeURIComponent(eventId)}&market=${encodeURIComponent(market)}&target=${currentTarget()}&threshold=${encodeURIComponent($("threshold").value || "0.10")}`
      );
      applyPrediction(predict.formatted);
      const path = predict.path || {};
      paintPrice(charts.kalshi, path.kalshi, path.polymarket, KALSHI_COLOR, POLY_COLOR, false);
      paintPrice(charts.polymarket, path.polymarket, path.kalshi, POLY_COLOR, KALSHI_COLOR, false);
      if (charts.prob) {
        const lastProb = lastValue(path.prob_up);
        charts.prob.price.applyOptions({
          color: lastProb != null && lastProb >= 0.5 ? UP_COLOR : DOWN_COLOR,
        });
        charts.prob.price.setData(path.prob_up || []);
        charts.prob.chart.timeScale().fitContent();
        setLegend("prob-legend", `P(up) ${fmtPct(lastProb)}  ·  ${predict.formatted.target}`);
      }
      $("trading-volume").textContent = predict.volume_label || series.volume_label || "N/A";
    } else {
      applyPrediction(null);
      $("trading-volume").textContent = series.volume_label || "N/A";
    }

    status.textContent = `Updated ${new Date().toLocaleTimeString()} · ${series.kalshi_points} Kalshi pts · ${series.polymarket_points} Poly pts`;
  } catch (err) {
    status.textContent = err.message;
    if ($("algo-select").value === "xgboost") applyPrediction(null);
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(refresh, 20000);
}

document.addEventListener("DOMContentLoaded", async () => {
  window.addEventListener("resize", () => {
    resizeChart(charts.kalshi);
    resizeChart(charts.polymarket);
    resizeChart(charts.prob);
  });

  $("event-select").addEventListener("change", loadMarkets);
  $("market-select").addEventListener("change", () => {
    refresh();
    startPolling();
  });
  $("input-form").addEventListener("submit", (event) => {
    event.preventDefault();
    refresh();
    startPolling();
  });

  try {
    await loadEvents();
    await refresh();
    startPolling();
  } catch (err) {
    $("status-line").textContent = err.message;
  }
});
