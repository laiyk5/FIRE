/**
 * FIRE Simulator — Prediction Models
 * Provides fitting functions for salary, cost, and return-rate models.
 */

/* eslint-disable no-unused-vars */
const Models = (() => {
  // ────────────────────────────────────────────────────────────────────────────
  // Helpers
  // ────────────────────────────────────────────────────────────────────────────

  /** Logistic sigmoid σ(z) = 1 / (1 + e^{-z}) */
  function sigmoid(z) {
    // Clamp to avoid Infinity from very large |z|
    const clamped = Math.max(-500, Math.min(500, z));
    return 1 / (1 + Math.exp(-clamped));
  }

  // ────────────────────────────────────────────────────────────────────────────
  // SALARY MODELS
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * Logistic Salary Model fitted via Bayesian MAP estimation.
   *
   * MODEL: f(year) = L · σ(k · (year − m))
   *   where m = inflection year (salary growth is fastest),
   *         L = salary ceiling,
   *         k = steepness (per year).
   *
   * TIME: raw calendar years (no normalisation).  Salaries are normalised by
   * maxVal so gradients for Ln = L/maxVal stay at O(1) regardless of salary
   * magnitude.  k is O(0.05 /year) and m is O(10 years from minYear), so each
   * parameter gets its own learning rate.
   *
   * ADAPTIVE PRIORS anchored to the full projection horizon:
   *   priorM  = 0.3 · totalSpan  (inflection 30 % through the full career arc)
   *   priorLn = 2.5              (ceiling ≈ 2.5× current max salary)
   *   priorK  derived analytically so that f(lastHistYear) ≈ maxSal:
   *            sigmoid(priorK · (lastHistYear − minYear − priorM)) = 1/priorLn
   *            → inflection is always in the future, model anchors to last data point
   *
   * BAYESIAN REGULARISATION: dividing the data term by n means the prior
   * weakens as more observations arrive — the defining property of Bayesian
   * MAP estimation.
   *
   *   -log P(θ|data) ∝ (1/n)·Σ(f_i − ŷ_i)²  +  λLn·(Ln−priorLn)²
   *                                            +  λK·(k−priorK)²
   *                                            +  λM·(m−priorM)²
   *
   * BOUNDS: m ≥ lastHistYear − minYear + 1  (inflection always in the future).
   *
   * @param {number[]} years    — sorted historical years
   * @param {number[]} salaries — historical salaries aligned to years
   * @param {number}   endYear  — last predicted year (full horizon end)
   * @returns {(year: number) => number}
   */
  function fitLogisticLinear(years, salaries, endYear) {
    if (years.length === 0) return () => 0;
    if (years.length === 1) return () => salaries[0];

    const n           = years.length;
    const minYear     = years[0];
    const lastHistYear = years[n - 1];
    const histSpan    = Math.max(lastHistYear - minYear, 1);
    // endYear is passed by the caller; the fallback is a defensive safety net only.
    const predictYrs  = Math.max((endYear || lastHistYear + 30) - lastHistYear, 1);
    const totalSpan   = histSpan + predictYrs;

    const maxVal = Math.max(...salaries) || 1;
    const ys     = salaries.map(s => s / maxVal); // normalised salaries ∈ [0, 1]

    // ── Adaptive Bayesian priors ──────────────────────────────────────────────
    // priorM: inflection 30 % into the full career arc (in years from minYear).
    // This pushes the S-curve peak well past the observed data so extrapolation
    // is smooth and gradual rather than saturating immediately after last year.
    const priorLn = 2.5;                               // ceiling at 2.5× current max
    const priorM  = 0.3 * totalSpan;                   // in calendar years from minYear

    // k derived so the model "passes through" the last historical salary:
    //   sigmoid(k · (histSpan − priorM)) = ys_last / priorLn
    // → k = logit(ys_last / priorLn) / (histSpan − priorM)
    // Since histSpan < priorM (inflection is in the future), denominator < 0
    // and k > 0 (logit of a value < 0.5 is negative).
    const denomK  = histSpan - priorM;                 // < 0 when inflection in future
    // Clamp the logit argument to (0.01, 0.99) to prevent NaN / ±Infinity when
    // the last observed salary is near or above the prior ceiling.
    const logitArg = Math.min(0.99, Math.max(0.01, ys[n - 1] / priorLn));
    const logitVal = Math.log(logitArg / (1 - logitArg));
    const priorK  = Math.max(0.001, denomK !== 0 ? logitVal / denomK : 0.05);

    // Precision λ = 1/σ²
    const λLn = 1 / (0.80 ** 2);                         // moderate ceiling prior
    // Cap λK so a tiny priorK (fallback 0.001) cannot freeze k at its prior value.
    const λK  = Math.min(1 / (priorK ** 2), 10000);
    const λM  = 1 / ((totalSpan * 0.15) ** 2);           // σM = 15 % of total span

    // ── MAP optimisation ──────────────────────────────────────────────────────
    // Separate learning rates because k (≈0.05/yr) and m (≈10 yrs) live at
    // very different scales; a single lr would be unstable for one of them.
    let Ln = priorLn, k = priorK, m = priorM;
    const lrLn = 0.01;    // Ln is O(1) in normalised salary units
    const lrK  = 0.0001;  // k  is O(0.05 /year) — small → needs small step
    const lrM  = 0.05;    // m  is O(10 years) from minYear

    for (let iter = 0; iter < 10000; iter++) {
      let gLn = 0, gK = 0, gM = 0;

      for (let i = 0; i < n; i++) {
        const t   = years[i] - minYear;               // calendar years from start
        const sig = sigmoid(k * (t - m));
        const err = Ln * sig - ys[i];

        gLn += err * sig;
        gK  += err * Ln * sig * (1 - sig) * (t - m);
        gM  += err * Ln * sig * (1 - sig) * (-k);
      }

      // MAP update: data gradient (normalised by n) + prior gradient
      Ln -= lrLn * (2 * gLn / n  +  2 * λLn * (Ln - priorLn));
      k  -= lrK  * (2 * gK  / n  +  2 * λK  * (k  - priorK));
      m  -= lrM  * (2 * gM  / n  +  2 * λM  * (m  - priorM));

      // Hard parameter constraints
      Ln = Math.max(1.0, Ln);                         // ceiling ≥ observed max
      k  = Math.max(0.001, Math.min(2.0, k));         // bounded steepness (per year)
      // m is in years from minYear; lastHistYear − minYear = histSpan,
      // so the constraint m ≥ histSpan + 1 places the inflection at least
      // 1 year after lastHistYear (i.e., always in the future).
      m  = Math.max(histSpan + 1, m);
    }

    const L = Ln * maxVal;
    const predict = (year) => {
      const t = year - minYear;                       // calendar years from start
      return Math.max(0, L * sigmoid(k * (t - m)));
    };
    return { predict, params: { L, k, m, minYear } };
  }

  // ────────────────────────────────────────────────────────────────────────────
  // COST MODELS
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * Fixed Save Rate: costs = salary × (1 − save_rate).
   * save_rate is estimated as mean(1 − costs_i / salary_i) over historical data.
   */
  function fitFixedSaveRate(years, salaries, costs) {
    if (salaries.length === 0) return () => 0;
    const rates = salaries.map((s, i) => (s > 0 ? 1 - costs[i] / s : 0));
    const avgSaveRate = rates.reduce((a, b) => a + b, 0) / rates.length;
    return (_year, salary) => Math.max(0, salary * (1 - avgSaveRate));
  }

  /**
   * Inflation + Linear: costs(t) = base · (1+r)^dt + slope · dt
   * r is estimated from compound growth; slope from residuals.
   */
  function fitInflationLinear(years, costs) {
    if (years.length === 0) return () => 0;
    if (years.length === 1) return () => costs[0];

    const t0 = years[0];
    const base = costs[0] || 1;
    const dtEnd = years[years.length - 1] - t0;

    const annualRate = dtEnd > 0 && base > 0
      ? Math.pow(costs[costs.length - 1] / base, 1 / dtEnd) - 1
      : 0.03;

    // Linear residual slope
    let numSlope = 0, denSlope = 0;
    for (let i = 0; i < years.length; i++) {
      const dt = years[i] - t0;
      const predicted = base * Math.pow(1 + annualRate, dt);
      const residual = costs[i] - predicted;
      numSlope += residual * dt;
      denSlope += dt * dt;
    }
    const slope = denSlope > 0 ? numSlope / denSlope : 0;

    return (year) => {
      const dt = year - t0;
      return Math.max(0, base * Math.pow(1 + annualRate, dt) + slope * dt);
    };
  }

  /**
   * Bell Curve: costs(t) = amplitude · exp(−(t−peak)²/(2σ²)) + base
   * Useful for costs that peak in middle age.
   */
  function fitBellCurve(years, costs) {
    if (years.length === 0) return () => 0;
    if (years.length === 1) return () => costs[0];

    const minCost = Math.min(...costs);
    const maxCost = Math.max(...costs);
    const peakYear = years[costs.indexOf(maxCost)];
    const sigma = Math.max((years[years.length - 1] - years[0]) / 2, 5);
    const amplitude = maxCost - minCost;

    return (year) => {
      const diff = year - peakYear;
      return Math.max(
        0,
        amplitude * Math.exp(-(diff * diff) / (2 * sigma * sigma)) + minCost
      );
    };
  }

  // ────────────────────────────────────────────────────────────────────────────
  // RETURN MODELS
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * 3-Year Moving Average: average of the last (up to) 3 historical return rates.
   */
  function fitMA3(_years, returnRates) {
    const lastN = returnRates.slice(-3);
    const avg = lastN.reduce((a, b) => a + b, 0) / (lastN.length || 1);
    return () => avg;
  }

  /**
   * Fixed Rate: always returns the user-specified rate.
   */
  function fitFixedRate(rate) {
    return () => rate;
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────────────────────────────────────
  return {
    /**
     * @param {'logisticLinear'} model
     * @param {number[]} years
     * @param {number[]} salaries
     * @param {number}   endYear  — last predicted year; used for full-horizon normalisation
     * @returns {{ predict: (year: number) => number, params: { L: number, k: number, m: number, minYear: number } }}
     */
    fitSalary(model, years, salaries, endYear) {
      switch (model) {
        case 'logisticLinear':
        default:
          return fitLogisticLinear(years, salaries, endYear);
      }
    },

    /**
     * @param {'fixedSaveRate'|'inflationLinear'|'bellCurve'} model
     * @param {number[]} years
     * @param {number[]} salaries
     * @param {number[]} costs
     * @returns {(year: number, salary: number) => number}
     */
    fitCosts(model, years, salaries, costs) {
      switch (model) {
        case 'inflationLinear':
          return fitInflationLinear(years, costs);
        case 'bellCurve':
          return fitBellCurve(years, costs);
        case 'fixedSaveRate':
        default:
          return fitFixedSaveRate(years, salaries, costs);
      }
    },

    /**
     * @param {'ma3'|'fixedRate'} model
     * @param {number[]} years
     * @param {number[]} returnRates  — as decimals (e.g. 0.07 = 7 %)
     * @param {number}   fixedRate    — decimal, used when model === 'fixedRate'
     * @returns {(year: number) => number}
     */
    fitReturn(model, years, returnRates, fixedRate) {
      switch (model) {
        case 'fixedRate':
          return fitFixedRate(fixedRate);
        case 'ma3':
        default:
          return fitMA3(years, returnRates);
      }
    },

    /**
     * Build a salary predictor directly from explicit logistic parameters.
     * Used when the user manually overrides L, k, or m in the UI.
     * @param {{ L: number, k: number, m: number, minYear: number }} params
     * @returns {(year: number) => number}
     */
    salaryPredictorFromParams({ L, k, m, minYear }) {
      return (year) => {
        const t = year - minYear;
        return Math.max(0, L * sigmoid(k * (t - m)));
      };
    },
  };
})();
