/**
 * FIRE Simulator — Alpine.js Application Logic
 *
 * Exposes a single factory function `fireApp()` consumed by the
 * `x-data` directive in index.html.
 */

/* global Models, Simulator, Charts */
/* eslint-disable no-unused-vars */

function fireApp() {
  return {
    // ── Initial Setup ────────────────────────────────────────────────────────
    currentYear:        new Date().getFullYear(),
    initialAssets:      50000,
    safeWithdrawalRate: 4,    // %
    predictYears:       30,

    // ── Historical Data ──────────────────────────────────────────────────────
    historicalData: [],

    // ── Model Selections ─────────────────────────────────────────────────────
    salaryModel:     'logisticLinear',
    costModel:       'fixedSaveRate',
    returnModel:     'ma3',
    fixedReturnRate: 7,       // % — used when returnModel === 'fixedRate'

    // ── Processed year entries (historical + predicted, kept mutable) ────────
    allYearsData: [],

    // ── Simulation results ───────────────────────────────────────────────────
    simulationResults: [],
    hasResults:        false,

    // ── Year selection & edit panel ──────────────────────────────────────────
    selectedYear: null,
    editData:     null,

    // ── Summary metrics ──────────────────────────────────────────────────────
    fireYear:      null,
    finalCoverage: 0,
    finalAssets:   0,

    // ────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ────────────────────────────────────────────────────────────────────────

    init() {
      const cy = this.currentYear;
      this.historicalData = [
        { year: cy - 3, salary: 70000, costs: 50000, returnRate: 8,  assetChanges: '' },
        { year: cy - 2, salary: 75000, costs: 52000, returnRate: 12, assetChanges: '' },
        { year: cy - 1, salary: 80000, costs: 54000, returnRate: 7,  assetChanges: '' },
      ];
    },

    // ────────────────────────────────────────────────────────────────────────
    // Historical Data Table
    // ────────────────────────────────────────────────────────────────────────

    addHistoricalYear() {
      const last = this.historicalData.length > 0
        ? this.historicalData[this.historicalData.length - 1]
        : { year: this.currentYear - 1, salary: 0, costs: 0, returnRate: 7 };
      this.historicalData.push({
        year:         last.year + 1,
        salary:       last.salary,
        costs:        last.costs,
        returnRate:   last.returnRate,
        assetChanges: '',
      });
    },

    removeHistoricalYear(index) {
      this.historicalData.splice(index, 1);
    },

    // ────────────────────────────────────────────────────────────────────────
    // Auto-Fit & Predict
    // ────────────────────────────────────────────────────────────────────────

    autoFitAndPredict() {
      // Sanitise & sort historical data
      const hist = [...this.historicalData]
        .sort((a, b) => a.year - b.year)
        .map(d => ({
          year:         parseInt(d.year)        || 0,
          salary:       parseFloat(d.salary)    || 0,
          costs:        parseFloat(d.costs)     || 0,
          returnRate:   parseFloat(d.returnRate)|| 0,
          assetChanges: parseFloat(d.assetChanges) || 0,
        }));

      if (hist.length === 0) {
        alert('Please add at least one row of historical data first.');
        return;
      }

      const years       = hist.map(d => d.year);
      const salaries    = hist.map(d => d.salary);
      const costs       = hist.map(d => d.costs);
      const returnRates = hist.map(d => d.returnRate / 100);

      // ── Fit models ────────────────────────────────────────────────────────
      const salaryPredictor = Models.fitSalary(this.salaryModel, years, salaries);
      const costPredictor   = Models.fitCosts(this.costModel, years, salaries, costs);
      const returnPredictor = Models.fitReturn(
        this.returnModel,
        years,
        returnRates,
        this.fixedReturnRate / 100,
      );

      // ── Build allYearsData ─────────────────────────────────────────────────
      this.allYearsData = [];

      for (const d of hist) {
        this.allYearsData.push({
          year:         d.year,
          salary:       d.salary,
          costs:        d.costs,
          returnRate:   d.returnRate / 100,
          assetChanges: d.assetChanges,
          isHistorical: true,
        });
      }

      const lastHistYear = Math.max(...years);
      const endYear      = lastHistYear + (parseInt(this.predictYears) || 30);

      for (let year = lastHistYear + 1; year <= endYear; year++) {
        const salary     = salaryPredictor(year);
        const returnRate = returnPredictor(year);
        const cost       = costPredictor(year, salary);
        this.allYearsData.push({
          year,
          salary:       Math.round(salary),
          costs:        Math.round(cost),
          returnRate,
          assetChanges: 0,
          isHistorical: false,
        });
      }

      // ── Simulate ───────────────────────────────────────────────────────────
      this._runSimulation();
      this.hasResults  = true;
      this.selectedYear = null;
      this.editData     = null;

      this.$nextTick(() => {
        Charts.initOrUpdateCharts(this.simulationResults, this.selectedYear);
      });
    },

    // ────────────────────────────────────────────────────────────────────────
    // Simulation (internal — re-runs from allYearsData)
    // ────────────────────────────────────────────────────────────────────────

    _runSimulation() {
      this.simulationResults = Simulator.run({
        initialAssets: parseFloat(this.initialAssets) || 0,
        yearsData:     this.allYearsData,
        swr:           (parseFloat(this.safeWithdrawalRate) || 4) / 100,
      });

      const fireResult = this.simulationResults.find(r => r.isFIRE);
      this.fireYear = fireResult ? fireResult.year : null;

      const last = this.simulationResults[this.simulationResults.length - 1];
      this.finalCoverage = last ? last.coverageRatio : 0;
      this.finalAssets   = last ? last.assets        : 0;
    },

    // ────────────────────────────────────────────────────────────────────────
    // Year Selection & Editing
    // ────────────────────────────────────────────────────────────────────────

    selectYear(yearStr) {
      if (!yearStr) {
        this.selectedYear = null;
        this.editData     = null;
        Charts.updateHighlight(this.simulationResults, null);
        return;
      }

      const year = parseInt(yearStr);
      this.selectedYear = year;

      const entry  = this.allYearsData.find(d => d.year === year);
      const result = this.simulationResults.find(r => r.year === year);

      if (entry && result) {
        this.editData = {
          year:         entry.year,
          salary:       Math.round(entry.salary),
          costs:        Math.round(entry.costs),
          returnRate:   parseFloat((entry.returnRate * 100).toFixed(2)),
          assetChanges: entry.assetChanges || 0,
          assets:       Math.round(result.assets),
          isHistorical: entry.isHistorical,
        };
      }

      Charts.updateHighlight(this.simulationResults, year);
    },

    /** Called on every keystroke in the edit panel for predicted years. */
    updateEditData() {
      if (!this.editData || this.editData.isHistorical) return;

      const year = this.selectedYear;
      const idx  = this.allYearsData.findIndex(d => d.year === year);
      if (idx === -1) return;

      this.allYearsData[idx] = {
        ...this.allYearsData[idx],
        salary:       parseFloat(this.editData.salary)       || 0,
        costs:        parseFloat(this.editData.costs)        || 0,
        returnRate:   (parseFloat(this.editData.returnRate)  || 0) / 100,
        assetChanges: parseFloat(this.editData.assetChanges) || 0,
      };

      this._runSimulation();

      // Refresh computed assets in edit panel
      const result = this.simulationResults.find(r => r.year === year);
      if (result) this.editData.assets = Math.round(result.assets);

      Charts.updateHighlight(this.simulationResults, this.selectedYear);
    },

    // ────────────────────────────────────────────────────────────────────────
    // Formatting helpers
    // ────────────────────────────────────────────────────────────────────────

    formatNumber(n) {
      if (n === null || n === undefined) return '0';
      const abs = Math.abs(n);
      if (abs >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
      if (abs >= 1_000)     return (n / 1_000).toFixed(0)     + 'k';
      return Math.round(n).toString();
    },

    get yearsToFIRE() {
      if (!this.fireYear) return '∞';
      const diff = this.fireYear - this.currentYear;
      return diff <= 0 ? 'Now!' : diff;
    },
  };
}
