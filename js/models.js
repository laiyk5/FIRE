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
   * Logistic + Linear Salary Model.
   * f(t) = L · σ(k · (t_norm − t0))
   * where t_norm = (year − minYear) / span  →  [0, 1] over historical range,
   * and can exceed 1 for future years (salary approaches ceiling L).
   *
   * Fitted with gradient-descent MSE minimisation.
   *
   * @param {number[]} years
   * @param {number[]} salaries
   * @returns {(year: number) => number}
   */
  function fitLogisticLinear(years, salaries) {
    if (years.length === 0) return () => 0;
    if (years.length === 1) return () => salaries[0];

    const n = years.length;
    const minYear = years[0];
    const span = (years[n - 1] - minYear) || 1;
    const tn = years.map(y => (y - minYear) / span); // normalised [0,1]

    const maxVal = Math.max(...salaries) || 1;

    // ── Initial parameter estimates ──────────────────────────────────────────
    let L = maxVal * 1.5;   // ceiling
    let k = 4.0;            // steepness in normalised time
    let t0 = 0.5;           // midpoint in normalised time

    // ── Gradient descent ─────────────────────────────────────────────────────
    const lr = 0.01;
    for (let iter = 0; iter < 4000; iter++) {
      let gL = 0, gk = 0, gt0 = 0;
      for (let i = 0; i < n; i++) {
        const t = tn[i];
        const y = salaries[i];
        const z = k * (t - t0);
        const sig = sigmoid(z);
        const f = L * sig;
        const err = f - y;                        // residual

        gL  += err * sig;
        gk  += err * L * sig * (1 - sig) * (t - t0);
        gt0 += err * L * sig * (1 - sig) * (-k);
      }
      L  -= lr * 2 * gL  / n;
      k  -= lr * 2 * gk  / n;
      t0 -= lr * 2 * gt0 / n;

      // Soft constraints
      L  = Math.max(maxVal, L);
      k  = Math.max(0.01, Math.min(20, k));
    }

    return (year) => {
      const t = (year - minYear) / span;
      return Math.max(0, L * sigmoid(k * (t - t0)));
    };
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
     * @returns {(year: number) => number}
     */
    fitSalary(model, years, salaries) {
      switch (model) {
        case 'logisticLinear':
        default:
          return fitLogisticLinear(years, salaries);
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
  };
})();
