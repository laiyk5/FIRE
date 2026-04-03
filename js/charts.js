/**
 * FIRE Simulator — Chart Management
 *
 * Manages two Chart.js charts:
 *   • mainChart     — Salary / Costs / Capital Income / Net Worth over time
 *   • coverageChart — Two-Phase FIRE coverage over time
 *
 * Relies on Chart.js v4 and chartjs-plugin-annotation v3 loaded globally.
 */

/* eslint-disable no-unused-vars */
const Charts = (() => {
  // ── Colour palette ──────────────────────────────────────────────────────────
  const AMBER  = 'rgb(245, 158, 11)';
  const BLUE   = 'rgb(59, 130, 246)';
  const RED    = 'rgb(239, 68, 68)';
  const GREEN  = 'rgb(34, 197, 94)';
  const PURPLE = 'rgb(168, 85, 247)';
  const TEAL   = 'rgb(20, 184, 166)';

  let mainChart     = null;
  let coverageChart = null;

  // ── Helpers ─────────────────────────────────────────────────────────────────

  /** Format large numbers compactly for axis tick labels */
  function fmtK(v) {
    const abs = Math.abs(v);
    if (abs >= 1_000_000) return (v / 1_000_000).toFixed(1) + 'M';
    if (abs >= 1_000)     return (v / 1_000).toFixed(0)     + 'k';
    return Math.round(v).toString();
  }

  /** Build per-point colour array; selected year gets amber */
  function pointColours(results, selectedYear, defaultColour) {
    return results.map(r => r.year === selectedYear ? AMBER : defaultColour);
  }

  /** Build per-point radius array; selected year gets larger dot */
  function pointRadii(results, selectedYear) {
    return results.map(r => r.year === selectedYear ? 8 : 4);
  }

  /**
   * Segment callback for dashed predicted-year lines.
   * Segments starting at index ≥ histCount − 1 are dashed.
   */
  function segmentDash(histCount) {
    return {
      borderDash: (ctx) =>
        ctx.p0DataIndex >= histCount - 1 ? [6, 4] : [],
    };
  }

  /** Build annotation object for the selected-year vertical line */
  function selectedLineAnnotation(selectedYear) {
    if (!selectedYear) return {};
    return {
      selectedLine: {
        type: 'line',
        xMin: selectedYear,
        xMax: selectedYear,
        borderColor: AMBER,
        borderWidth: 2,
        borderDash: [4, 4],
        label: {
          content: String(selectedYear),
          display: true,
          position: 'start',
          backgroundColor: 'rgba(245,158,11,0.15)',
          color: AMBER,
          font: { weight: 'bold' },
        },
      },
    };
  }

  /** Shared horizontal threshold line at coverage = 1× */
  function fireLine() {
    return {
      type: 'line',
      yMin: 1, yMax: 1,
      borderColor: GREEN,
      borderWidth: 2,
      borderDash: [6, 3],
      label: {
        content: '🎯 FIRE threshold',
        display: true,
        position: 'end',
        backgroundColor: 'rgba(34,197,94,0.15)',
        color: GREEN,
        font: { weight: 'bold' },
      },
    };
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * (Re-)create both charts from scratch with the provided simulation results.
   * Call this after the first Auto-Fit & Predict, or whenever model changes.
   */
  function initOrUpdateCharts(results, selectedYear) {
    if (!results || results.length === 0) return;

    const mainCtx     = document.getElementById('mainChart');
    const coverageCtx = document.getElementById('coverageChart');
    if (!mainCtx || !coverageCtx) return;

    if (mainChart)     { mainChart.destroy();     mainChart     = null; }
    if (coverageChart) { coverageChart.destroy(); coverageChart = null; }

    const histCount     = results.filter(r => r.isHistorical).length;
    const labels        = results.map(r => r.year);
    const salaries      = results.map(r => r.salary);
    const costs         = results.map(r => r.costs);
    const capitalIncomes = results.map(r => r.capitalIncome);
    const assets        = results.map(r => r.assets);
    const phase1        = results.map(r => r.phase1Coverage);
    const phase2        = results.map(r => r.phase2Coverage);

    const radii  = pointRadii(results, selectedYear);
    const annots = selectedLineAnnotation(selectedYear);

    // ── Main chart (Salary / Costs / Capital Income / Net Worth) ────────────
    mainChart = new Chart(mainCtx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Salary',
            data: salaries,
            borderColor: BLUE,
            backgroundColor: 'rgba(59,130,246,0.05)',
            pointBackgroundColor: pointColours(results, selectedYear, BLUE),
            pointRadius: radii,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            fill: false,
            segment: segmentDash(histCount),
          },
          {
            label: 'Costs',
            data: costs,
            borderColor: RED,
            backgroundColor: 'rgba(239,68,68,0.05)',
            pointBackgroundColor: pointColours(results, selectedYear, RED),
            pointRadius: radii,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            fill: false,
            segment: segmentDash(histCount),
          },
          {
            label: 'Capital Income',
            data: capitalIncomes,
            borderColor: TEAL,
            backgroundColor: 'rgba(20,184,166,0.05)',
            pointBackgroundColor: pointColours(results, selectedYear, TEAL),
            pointRadius: radii,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            fill: false,
            segment: segmentDash(histCount),
          },
          {
            label: 'Net Worth',
            data: assets,
            borderColor: GREEN,
            backgroundColor: 'rgba(34,197,94,0.05)',
            pointBackgroundColor: pointColours(results, selectedYear, GREEN),
            pointRadius: radii,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            fill: false,
            yAxisID: 'y1',
            segment: segmentDash(histCount),
          },
        ],
      },
      options: {
        responsive: true,
        animation: { duration: 300 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          annotation: { annotations: annots },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: $${fmtK(ctx.parsed.y)}`,
            },
          },
        },
        scales: {
          x: { title: { display: true, text: 'Year' } },
          y: {
            title: { display: true, text: 'Amount ($)' },
            ticks: { callback: v => '$' + fmtK(v) },
          },
          y1: {
            position: 'right',
            title: { display: true, text: 'Net Worth ($)' },
            grid: { drawOnChartArea: false },
            ticks: { callback: v => '$' + fmtK(v) },
          },
        },
      },
    });

    // ── Coverage chart (Two-Phase FIRE) ─────────────────────────────────────
    coverageChart = new Chart(coverageCtx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Phase 1 — Income / Costs',
            data: phase1,
            borderColor: GREEN,
            backgroundColor: 'rgba(34,197,94,0.08)',
            pointBackgroundColor: pointColours(results, selectedYear, GREEN),
            pointRadius: radii,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            fill: false,
            segment: segmentDash(histCount),
          },
          {
            label: 'Phase 2 — Income / Salary',
            data: phase2,
            borderColor: PURPLE,
            backgroundColor: 'rgba(168,85,247,0.08)',
            pointBackgroundColor: pointColours(results, selectedYear, PURPLE),
            pointRadius: radii,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            fill: false,
            segment: segmentDash(histCount),
          },
        ],
      },
      options: {
        responsive: true,
        animation: { duration: 300 },
        plugins: {
          annotation: {
            annotations: {
              ...annots,
              fireLine: fireLine(),
            },
          },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}×`,
            },
          },
        },
        scales: {
          x: { title: { display: true, text: 'Year' } },
          y: {
            min: 0,
            title: { display: true, text: 'Coverage Ratio' },
            ticks: { callback: v => v.toFixed(1) + '×' },
          },
        },
      },
    });
  }

  /**
   * Incrementally update chart data and year highlight without rebuilding.
   * Faster than initOrUpdateCharts — used during live editing.
   */
  function updateHighlight(results, selectedYear) {
    if (!mainChart || !coverageChart) {
      initOrUpdateCharts(results, selectedYear);
      return;
    }

    const histCount      = results.filter(r => r.isHistorical).length;
    const labels         = results.map(r => r.year);
    const salaries       = results.map(r => r.salary);
    const costs          = results.map(r => r.costs);
    const capitalIncomes = results.map(r => r.capitalIncome);
    const assets         = results.map(r => r.assets);
    const phase1         = results.map(r => r.phase1Coverage);
    const phase2         = results.map(r => r.phase2Coverage);
    const radii          = pointRadii(results, selectedYear);
    const annots         = selectedLineAnnotation(selectedYear);

    // ── Update main chart ───────────────────────────────────────────────────
    mainChart.data.labels = labels;

    mainChart.data.datasets[0].data = salaries;
    mainChart.data.datasets[0].pointBackgroundColor = pointColours(results, selectedYear, BLUE);
    mainChart.data.datasets[0].pointRadius = radii;
    mainChart.data.datasets[0].segment = segmentDash(histCount);

    mainChart.data.datasets[1].data = costs;
    mainChart.data.datasets[1].pointBackgroundColor = pointColours(results, selectedYear, RED);
    mainChart.data.datasets[1].pointRadius = radii;
    mainChart.data.datasets[1].segment = segmentDash(histCount);

    mainChart.data.datasets[2].data = capitalIncomes;
    mainChart.data.datasets[2].pointBackgroundColor = pointColours(results, selectedYear, TEAL);
    mainChart.data.datasets[2].pointRadius = radii;
    mainChart.data.datasets[2].segment = segmentDash(histCount);

    mainChart.data.datasets[3].data = assets;
    mainChart.data.datasets[3].pointBackgroundColor = pointColours(results, selectedYear, GREEN);
    mainChart.data.datasets[3].pointRadius = radii;
    mainChart.data.datasets[3].segment = segmentDash(histCount);

    mainChart.options.plugins.annotation.annotations = annots;
    mainChart.update('none');

    // ── Update coverage chart ───────────────────────────────────────────────
    coverageChart.data.labels = labels;

    coverageChart.data.datasets[0].data = phase1;
    coverageChart.data.datasets[0].pointBackgroundColor = pointColours(results, selectedYear, GREEN);
    coverageChart.data.datasets[0].pointRadius = radii;
    coverageChart.data.datasets[0].segment = segmentDash(histCount);

    coverageChart.data.datasets[1].data = phase2;
    coverageChart.data.datasets[1].pointBackgroundColor = pointColours(results, selectedYear, PURPLE);
    coverageChart.data.datasets[1].pointRadius = radii;
    coverageChart.data.datasets[1].segment = segmentDash(histCount);

    coverageChart.options.plugins.annotation.annotations = {
      ...annots,
      fireLine: fireLine(),
    };
    coverageChart.update('none');
  }

  return { initOrUpdateCharts, updateHighlight };
})();
