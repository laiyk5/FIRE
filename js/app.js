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
    currentYear:   new Date().getFullYear(),
    initialAssets: 50000,
    predictYears:  30,

    // ── Historical Data ──────────────────────────────────────────────────────
    historicalData: [],

    // ── Model Selections ─────────────────────────────────────────────────────
    salaryModel:     'careerGamma',
    costModel:       'fixedSaveRate',
    returnModel:     'ma3',
    fixedReturnRate: 7,       // % — used when returnModel === 'fixedRate'
    salaryAlpha:     0.8,     // prior stiffness for CareerGamma (0 = data only, 1 = balanced)
    salaryExpOffset: 0,       // years of experience before first data year (CareerGamma x-axis shift)

    // ── Fitted salary model parameters (exposed for manual override) ─────────
    salaryParams:       null,   // set after autoFitAndPredict; shape depends on model
    // Logistic params
    manualSalaryL:      null,   // logistic: salary ceiling ($)
    manualSalaryK:      null,   // logistic: steepness (1/yr) | gamma: growth exponent
    manualSalaryM:      null,   // logistic: inflection offset (yrs from minYear)
    // Gamma params
    manualSalaryA:      null,   // gamma: scale parameter
    manualSalaryB:      null,   // gamma: decay rate (per exp-year)
    manualSalaryC:      null,   // gamma: salary floor ($)

    // ── Fitted cost model parameters (exposed for manual override) ───────────
    costParams:            null,  // set after autoFitAndPredict
    manualSaveRate:        null,  // fixedSaveRate: save rate (%)
    manualInflationRate:   null,  // inflationLinear: annual rate (%)
    manualInflationSlope:  null,  // inflationLinear: linear slope ($/yr)
    manualBellAmplitude:   null,  // bellCurve: amplitude ($)
    manualBellPeakYear:    null,  // bellCurve: peak year
    manualBellSigma:       null,  // bellCurve: half-width (years)
    manualBellBase:        null,  // bellCurve: base cost floor ($)

    // ── Fitted return model parameters (exposed for manual override) ─────────
    returnParams:      null,  // set after autoFitAndPredict
    manualReturnRate:  null,  // % — computed avg for ma3 or user value for fixedRate

    // ── Processed year entries (historical + predicted, kept mutable) ────────
    allYearsData: [],

    // ── Simulation results ───────────────────────────────────────────────────
    simulationResults: [],
    hasResults:        false,

    // ── Year selection & edit panel ──────────────────────────────────────────
    selectedYear: null,
    editData:     null,

    // ── Summary metrics (Two-Phase FIRE) ─────────────────────────────────────
    firePhase1Year:    null,   // capital income ≥ costs
    firePhase2Year:    null,   // capital income ≥ salary
    finalPhase1Cover:  0,
    finalPhase2Cover:  0,
    finalAssets:       0,

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

      const lastHistYear = Math.max(...years);
      const endYear      = lastHistYear + (parseInt(this.predictYears) || 30);

      // ── Fit models ────────────────────────────────────────────────────────
      const { predict: salaryPredictor, params: salaryParams } =
        Models.fitSalary(this.salaryModel, years, salaries, endYear, { alpha: this.salaryAlpha, expOffset: this.salaryExpOffset });
      this.salaryParams = salaryParams;

      // Initialise manual parameter fields based on which model was fitted
      if (salaryParams.model === 'careerGamma') {
        this.manualSalaryA = parseFloat(salaryParams.a.toFixed(2));
        this.manualSalaryK = parseFloat(salaryParams.k.toFixed(4));
        this.manualSalaryB = parseFloat(salaryParams.b.toFixed(4));
        this.manualSalaryC = parseFloat(salaryParams.c.toFixed(2));
      } else {
        this.manualSalaryL = Math.round(salaryParams.L);
        this.manualSalaryK = parseFloat(salaryParams.k.toFixed(4));
        this.manualSalaryM = parseFloat(salaryParams.m.toFixed(2));
      }
      const costResult   = Models.fitCosts(this.costModel, years, salaries, costs);
      const returnResult = Models.fitReturn(
        this.returnModel,
        years,
        returnRates,
        this.fixedReturnRate / 100,
      );
      const costPredictor   = costResult.predict;
      const returnPredictor = returnResult.predict;

      // Store cost params and initialise manual fields
      this.costParams = costResult.params;
      const cp = costResult.params;
      if (cp.model === 'fixedSaveRate') {
        this.manualSaveRate = this._toPercent(cp.saveRate, 2);
      } else if (cp.model === 'inflationLinear') {
        this.manualInflationRate  = this._toPercent(cp.annualRate, 2);
        this.manualInflationSlope = parseFloat(cp.slope.toFixed(2));
      } else if (cp.model === 'bellCurve') {
        this.manualBellAmplitude = parseFloat(cp.amplitude.toFixed(2));
        this.manualBellPeakYear  = cp.peakYear;
        this.manualBellSigma     = parseFloat(cp.sigma.toFixed(2));
        this.manualBellBase      = parseFloat(cp.base.toFixed(2));
      }

      // Store return params and initialise manual field
      this.returnParams     = returnResult.params;
      this.manualReturnRate = this._toPercent(returnResult.params.rate, 4);

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
      this.hasResults   = true;
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
      });

      // Phase 1: first year where capital income ≥ costs
      const p1 = this.simulationResults.find(r => r.isPhase1FIRE);
      this.firePhase1Year = p1 ? p1.year : null;

      // Phase 2: first year where capital income ≥ salary
      const p2 = this.simulationResults.find(r => r.isPhase2FIRE);
      this.firePhase2Year = p2 ? p2.year : null;

      const last = this.simulationResults[this.simulationResults.length - 1];
      this.finalPhase1Cover = last ? last.phase1Coverage : 0;
      this.finalPhase2Cover = last ? last.phase2Coverage : 0;
      this.finalAssets      = last ? last.assets         : 0;
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
          capitalIncome: Math.round(result.capitalIncome),
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

      // Refresh computed fields in edit panel
      const result = this.simulationResults.find(r => r.year === year);
      if (result) {
        this.editData.assets       = Math.round(result.assets);
        this.editData.capitalIncome = Math.round(result.capitalIncome);
      }

      Charts.updateHighlight(this.simulationResults, this.selectedYear);
    },

    /**
     * Rebuild all predicted salaries using the manually adjusted model parameters,
     * then re-run the simulation so all charts and FIRE metrics update at once.
     */
    reapplySalaryParams() {
      if (!this.salaryParams) return;

      let params;
      if (this.salaryParams.model === 'careerGamma') {
        const sp = this.salaryParams;
        params = {
          ...sp,
          a: isFinite(parseFloat(this.manualSalaryA)) ? parseFloat(this.manualSalaryA) : sp.a,
          k: isFinite(parseFloat(this.manualSalaryK)) ? parseFloat(this.manualSalaryK) : sp.k,
          b: isFinite(parseFloat(this.manualSalaryB)) ? parseFloat(this.manualSalaryB) : sp.b,
          c: isFinite(parseFloat(this.manualSalaryC)) ? parseFloat(this.manualSalaryC) : sp.c,
        };
      } else {
        params = {
          ...this.salaryParams,
          L: parseFloat(this.manualSalaryL) || this.salaryParams.L,
          k: parseFloat(this.manualSalaryK) || this.salaryParams.k,
          m: parseFloat(this.manualSalaryM) || this.salaryParams.m,
        };
      }

      const predictor = Models.salaryPredictorFromParams(params);
      const histEntries = this.allYearsData.filter(d => d.isHistorical);
      const { predict: costPredictor } = Models.fitCosts(
        this.costModel,
        histEntries.map(d => d.year),
        histEntries.map(d => d.salary),
        histEntries.map(d => d.costs),
      );

      for (const entry of this.allYearsData) {
        if (!entry.isHistorical) {
          const salary = predictor(entry.year);
          entry.salary = Math.round(salary);
          entry.costs  = Math.round(costPredictor(entry.year, salary));
        }
      }

      this._runSimulation();
      Charts.updateHighlight(this.simulationResults, this.selectedYear);
    },

    /**
     * Rebuild all predicted costs using the manually adjusted cost model parameters,
     * then re-run the simulation so all charts and FIRE metrics update at once.
     */
    reapplyCostParams() {
      if (!this.costParams) return;

      let params;
      const cp = this.costParams;
      if (cp.model === 'fixedSaveRate') {
        const sr = this._parsePercentOrDefault(this.manualSaveRate, cp.saveRate);
        params = { ...cp, saveRate: sr };
      } else if (cp.model === 'inflationLinear') {
        params = {
          ...cp,
          annualRate: this._parsePercentOrDefault(this.manualInflationRate,  cp.annualRate),
          slope:      this._parseOrDefault(this.manualInflationSlope, cp.slope),
        };
      } else if (cp.model === 'bellCurve') {
        params = {
          ...cp,
          amplitude: this._parseOrDefault(this.manualBellAmplitude, cp.amplitude),
          peakYear:  this._parseOrDefault(this.manualBellPeakYear,  cp.peakYear),
          sigma:     this._parseOrDefault(this.manualBellSigma,     cp.sigma),
          base:      this._parseOrDefault(this.manualBellBase,      cp.base),
        };
      } else {
        return;
      }

      const costPredictor = Models.costPredictorFromParams(params);

      for (const entry of this.allYearsData) {
        if (!entry.isHistorical) {
          entry.costs = Math.round(costPredictor(entry.year, entry.salary));
        }
      }

      this._runSimulation();
      Charts.updateHighlight(this.simulationResults, this.selectedYear);
    },

    /**
     * Rebuild all predicted return rates using the manually adjusted return rate,
     * then re-run the simulation so all charts and FIRE metrics update at once.
     */
    reapplyReturnParams() {
      if (!this.returnParams) return;

      const rate = this._parsePercentOrDefault(this.manualReturnRate, this.returnParams.rate);

      const params = { ...this.returnParams, rate };
      const returnPredictor = Models.returnPredictorFromParams(params);

      for (const entry of this.allYearsData) {
        if (!entry.isHistorical) {
          entry.returnRate = returnPredictor(entry.year);
        }
      }

      this._runSimulation();
      Charts.updateHighlight(this.simulationResults, this.selectedYear);
    },

    // ────────────────────────────────────────────────────────────────────────
    // Formatting helpers
    // ────────────────────────────────────────────────────────────────────────

    /** Convert a decimal fraction to a rounded percentage string (e.g. 0.075 → "7.5000"). */
    _toPercent(value, decimals) {
      return parseFloat((value * 100).toFixed(decimals));
    },

    /**
     * Parse a user-entered percentage string to a decimal, falling back to the given default.
     * e.g. parsePercentOrDefault("7.5", 0.08) → 0.075
     */
    _parsePercentOrDefault(str, fallback) {
      const v = parseFloat(str);
      return isFinite(v) ? v / 100 : fallback;
    },

    /** Parse a numeric string, falling back to the given default if invalid. */
    _parseOrDefault(str, fallback) {
      const v = parseFloat(str);
      return isFinite(v) ? v : fallback;
    },

    formatNumber(n) {
      if (n === null || n === undefined) return '0';
      const abs = Math.abs(n);
      if (abs >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
      if (abs >= 1_000)     return (n / 1_000).toFixed(0)     + 'k';
      return Math.round(n).toString();
    },

    get yearsToPhase1() {
      if (!this.firePhase1Year) return '∞';
      const diff = this.firePhase1Year - this.currentYear;
      return diff <= 0 ? 'Now!' : diff;
    },

    get yearsToPhase2() {
      if (!this.firePhase2Year) return '∞';
      const diff = this.firePhase2Year - this.currentYear;
      return diff <= 0 ? 'Now!' : diff;
    },
  };
}
