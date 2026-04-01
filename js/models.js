/**
 * FIRE Simulator — Prediction Models
 * Provides fitting functions for salary, cost, and return-rate models.
 */

/* eslint-disable no-unused-vars */
const Models = (() => {
  // ────────────────────────────────────────────────────────────────────────────
  // Helpers
  // ────────────────────────────────────────────────────────────────────────────

  /** Default annual inflation rate used when the data span is too small to estimate one. */
  const DEFAULT_INFLATION_RATE = 0.03;

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
    return { predict, params: { model: 'logisticLinear', L, k, m, minYear } };
  }

  // ────────────────────────────────────────────────────────────────────────────
  // CAREER GAMMA SALARY MODEL
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * China salary prior data (years of experience → salary normalized by career peak).
   *
   * Data points: entry (yr 1), junior (yr 6), mid (yr 11), senior/peak (yr 18),
   * late-career (yr 31). Raw values: 10, 20, 35, 42.5, 30 k RMB/month, representative
   * of typical Chinese tech/professional sector (Tier-1 cities, ~2015-2023 average).
   * All values are normalized by the peak (42.5) so the shape is scale-invariant.
   * Users whose career follows a different country or industry should lower alpha
   * (prior stiffness) to let their own data dominate the fit.
   */
  const GAMMA_PRIOR_YEARS = [1, 6, 11, 18, 31];
  const GAMMA_PRIOR_SALS  = [10 / 42.5, 20 / 42.5, 35 / 42.5, 1.0, 30 / 42.5];

  /**
   * Large finite penalty returned by gammaEval when the power term overflows.
   * Distinct from the 1e10 "invalid parameter" penalty in objective functions:
   * 1e200 signals a numerical overflow in the function evaluation itself,
   * whereas 1e10 signals that a parameter is outside its valid domain.
   */
  const GAMMA_OVERFLOW_PENALTY = 1e200;

  /**
   * Small floor value used to prevent division-by-zero when normalising
   * salary or prior values.
   */
  const GAMMA_NORM_FLOOR = 1e-10;

  /**
   * Upper bound (inclusive) for the career-year search during auto-estimation
   * of expOffset.  35 covers the typical working life (≈ 35 years from entry
   * to late career) and matches the extent of the China prior data (x = 31).
   */
  const GAMMA_MAX_CAREER_SEARCH = 35;

  /**
   * Evaluate the Career Gamma function: f(x) = a · x^k · exp(−b·x) + c
   * where x = years of experience (x > 0, 1-indexed so x=1 is first career year).
   */
  function gammaEval(x, a, k, b, c) {
    const sx = Math.max(x, 0.001);
    const bx = b * sx;
    if (bx > 700) return Math.max(0, c);           // exp underflows to 0
    const xk = Math.pow(sx, k);
    if (!isFinite(xk)) return GAMMA_OVERFLOW_PENALTY;
    return a * xk * Math.exp(-bx) + c;
  }

  /**
   * Nelder-Mead derivative-free optimizer.
   * Minimises fn(params) starting from x0; returns best parameter vector found.
   * Tolerance of 1e-8 is sufficient for salary fitting where values are in thousands.
   * @param {(p: number[]) => number} fn
   * @param {number[]} x0
   * @param {number} [maxIter=10000]
   * @returns {number[]}
   */
  function nelderMead(fn, x0, maxIter) {
    maxIter = maxIter || 10000;
    const n = x0.length;
    const tol = 1e-8;

    // Build initial simplex: one vertex per dimension displaced by 5 %
    const verts = [x0.slice()];
    for (let i = 0; i < n; i++) {
      const v = x0.slice();
      v[i] += Math.abs(v[i]) > 1e-8 ? 0.05 * Math.abs(v[i]) : 0.00025;
      verts.push(v);
    }
    let fvals = verts.map(v => fn(v));

    for (let iter = 0; iter < maxIter; iter++) {
      // Sort: best (lowest) first
      const ord = fvals.map((f, i) => [f, i]).sort((a, b) => a[0] - b[0]);
      const sv  = ord.map(o => verts[o[1]].slice());
      const sf  = ord.map(o => fvals[o[1]]);

      if (sf[n] - sf[0] < tol) break;

      // Centroid of all but worst
      const c = new Array(n).fill(0);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          c[j] += sv[i][j] / n;

      // Reflection
      const xr = c.map((cj, j) => 2 * cj - sv[n][j]);
      const fr  = fn(xr);

      if (fr < sf[0]) {
        // Expansion
        const xe = c.map((cj, j) => 3 * cj - 2 * sv[n][j]);
        const fe  = fn(xe);
        sv[n] = fe < fr ? xe : xr;
        sf[n] = Math.min(fe, fr);
      } else if (fr < sf[n - 1]) {
        sv[n] = xr; sf[n] = fr;
      } else {
        // Contraction
        const useRef = fr < sf[n];
        const xc = c.map((cj, j) => cj + 0.5 * ((useRef ? xr[j] : sv[n][j]) - cj));
        const fc  = fn(xc);
        if (fc < (useRef ? fr : sf[n])) {
          sv[n] = xc; sf[n] = fc;
        } else {
          // Shrink toward best
          for (let i = 1; i <= n; i++) {
            sv[i] = sv[0].map((v, j) => v + 0.5 * (sv[i][j] - v));
            sf[i] = fn(sv[i]);
          }
        }
      }

      for (let i = 0; i <= n; i++) { verts[i] = sv[i]; fvals[i] = sf[i]; }
    }

    let best = 0;
    for (let i = 1; i <= n; i++) if (fvals[i] < fvals[best]) best = i;
    return verts[best];
  }

  /**
   * Career Gamma Salary Model with Bayesian regularisation against a country prior.
   *
   * MODEL: f(x) = a · x^k · exp(−b·x) + c
   *   where x = years of experience (= calendarYear − minYear + 1),
   *         a = scale (overall income magnitude),
   *         k = growth exponent (steepness of early-career climb),
   *         b = decay rate ("35-year crisis" / skill obsolescence),
   *         c = salary floor (entry-level base).
   *
   * PRIOR: fit on China average professional salary data (normalised by career peak).
   *
   * POSTERIOR MAP objective:
   *   E(p) = mean_i[(f(x_i,p) − ŷ_i)²]  +  α · Σ_j[((p_j − prior_j)/(|prior_j|+ε))²]
   *
   * Both China data and user data are normalised by their respective salary maxima
   * before fitting so the shape parameters k and b are directly comparable.
   *
   * @param {number[]} years    — sorted historical calendar years
   * @param {number[]} salaries — historical salaries aligned to years
   * @param {number}   endYear  — last predicted calendar year (unused but kept for API parity)
   * @param {number}   [alpha=0.8] — prior stiffness (0 = data only, 1 = balanced)
   * @param {number}   [expOffset=0] — total years of experience completed before the first data year.
   *   The model uses 1-indexed experience years: x = (calYear − minYear + 1) + expOffset,
   *   so x=1 means "end of first career year." With expOffset=5, your first data year is
   *   treated as year 6 of your career (5 prior years + the current year).
   * @returns {{ predict: (year: number) => number, params: object }}
   */
  function fitCareerGamma(years, salaries, endYear, alpha, expOffset) {
    if (alpha === undefined) alpha = 0.8;
    if (expOffset === undefined) expOffset = 0;

    if (years.length === 0) {
      return {
        predict: () => 0,
        params: { model: 'careerGamma', a: 0, k: 2, b: 0.1, c: 0, minYear: 0, alpha, expOffset },
      };
    }
    if (years.length === 1) {
      return {
        predict: () => salaries[0],
        params: { model: 'careerGamma', a: salaries[0], k: 2, b: 0.1, c: 0, minYear: years[0], alpha, expOffset },
      };
    }

    const minYear  = years[0];
    const maxSal   = Math.max(...salaries) || 1;
    const normSals = salaries.map(s => s / maxSal);

    // ── Step 1: fit China prior on normalised data ────────────────────────────
    const cFloor = Math.min(...GAMMA_PRIOR_SALS);
    const priorObj = (p) => {
      const [pa, pk, pb, pc] = p;
      if (pa < 0 || pk < 0 || pb < 0) return 1e10;
      let loss = 0;
      for (let i = 0; i < GAMMA_PRIOR_YEARS.length; i++) {
        const d = gammaEval(GAMMA_PRIOR_YEARS[i], pa, pk, pb, pc) - GAMMA_PRIOR_SALS[i];
        loss += d * d;
      }
      return loss / GAMMA_PRIOR_YEARS.length;
    };
    const priorParams = nelderMead(priorObj, [0.05, 2.0, 0.1, cFloor]);

    // ── Auto-estimate expOffset when left at 0 ───────────────────────────────
    // Find the career starting year x₀ whose relative growth pattern in the
    // China prior best matches the user's salary growth pattern.  This places
    // the historical data at a career stage consistent with the prior's shape,
    // avoiding the failure mode where data is placed at x=1 (entry-level) when
    // the user already has a mid-career salary trajectory.
    if (expOffset === 0 && normSals.length >= 2) {
      const n = normSals.length;
      const ref0 = Math.max(normSals[0], GAMMA_NORM_FLOOR);
      const userGrowth = normSals.map(v => v / ref0);

      let bestX0 = 1;
      let bestMSE = Infinity;
      for (let x0 = 1; x0 <= GAMMA_MAX_CAREER_SEARCH; x0++) {
        const priorRef = Math.max(gammaEval(x0, ...priorParams), GAMMA_NORM_FLOOR);
        let mse = 0;
        for (let i = 0; i < n; i++) {
          const pg = gammaEval(x0 + i, ...priorParams) / priorRef;
          const d  = userGrowth[i] - pg;
          mse += d * d;
        }
        mse /= n;
        if (mse < bestMSE) { bestMSE = mse; bestX0 = x0; }
      }
      expOffset = bestX0 - 1;   // first data year → career year bestX0
    }

    // x = years of experience: 1 + expOffset for first data year, growing from there
    const expYears = years.map(y => y - minYear + 1 + expOffset);

    // ── Step 2: fit user posterior with Bayesian regularisation ──────────────
    const postObj = (p) => {
      const [pa, pk, pb, pc] = p;
      if (pa < 0 || pk < 0 || pb < 0) return 1e10;

      // Data likelihood term (normalised by n)
      let dataLoss = 0;
      for (let i = 0; i < expYears.length; i++) {
        const d = gammaEval(expYears[i], pa, pk, pb, pc) - normSals[i];
        dataLoss += d * d;
      }
      dataLoss /= expYears.length;

      // Prior penalty (scale-invariant via normalisation by prior magnitude)
      let penalty = 0;
      for (let j = 0; j < 4; j++) {
        const denom = Math.abs(priorParams[j]) + 1e-6;
        const d = (p[j] - priorParams[j]) / denom;
        penalty += d * d;
      }

      return dataLoss + alpha * penalty;
    };
    const postParams = nelderMead(postObj, priorParams.slice());
    const [an, kn, bn, cn] = postParams;

    // De-normalise salary dimensions (a and c scale with salary, k and b are shape)
    const a = an * maxSal;
    const k = kn;
    const b = bn;
    const c = cn * maxSal;

    const predict = (year) => {
      const x = year - minYear + 1 + expOffset;
      return Math.max(0, gammaEval(x, a, k, b, c));
    };

    return { predict, params: { model: 'careerGamma', a, k, b, c, minYear, alpha, expOffset } };
  }

  // ────────────────────────────────────────────────────────────────────────────
  // COST MODELS
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * Fixed Save Rate: costs = salary × (1 − save_rate).
   * save_rate is estimated as mean(1 − costs_i / salary_i) over historical data.
   */
  function fitFixedSaveRate(years, salaries, costs) {
    if (salaries.length === 0) {
      return { predict: () => 0, params: { model: 'fixedSaveRate', saveRate: 0 } };
    }
    const rates = salaries.map((s, i) => (s > 0 ? 1 - costs[i] / s : 0));
    const avgSaveRate = rates.reduce((a, b) => a + b, 0) / rates.length;
    return {
      predict: (_year, salary) => Math.max(0, salary * (1 - avgSaveRate)),
      params:  { model: 'fixedSaveRate', saveRate: avgSaveRate },
    };
  }

  /**
   * Inflation + Linear: costs(t) = base · (1+r)^dt + slope · dt
   * r is estimated from compound growth; slope from residuals.
   */
  function fitInflationLinear(years, costs) {
    if (years.length === 0) {
      return { predict: () => 0, params: { model: 'inflationLinear', base: 0, annualRate: DEFAULT_INFLATION_RATE, slope: 0, t0: 0 } };
    }
    if (years.length === 1) {
      return { predict: () => costs[0], params: { model: 'inflationLinear', base: costs[0], annualRate: DEFAULT_INFLATION_RATE, slope: 0, t0: years[0] } };
    }

    const t0 = years[0];
    const base = costs[0] || 1;
    const dtEnd = years[years.length - 1] - t0;

    const annualRate = dtEnd > 0 && base > 0
      ? Math.pow(costs[costs.length - 1] / base, 1 / dtEnd) - 1
      : DEFAULT_INFLATION_RATE;

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

    return {
      predict: (year) => {
        const dt = year - t0;
        return Math.max(0, base * Math.pow(1 + annualRate, dt) + slope * dt);
      },
      params: { model: 'inflationLinear', base, annualRate, slope, t0 },
    };
  }

  /**
   * Bell Curve: costs(t) = amplitude · exp(−(t−peak)²/(2σ²)) + base
   * Useful for costs that peak in middle age.
   */
  function fitBellCurve(years, costs) {
    if (years.length === 0) {
      return { predict: () => 0, params: { model: 'bellCurve', amplitude: 0, peakYear: 0, sigma: 5, base: 0 } };
    }
    if (years.length === 1) {
      return { predict: () => costs[0], params: { model: 'bellCurve', amplitude: 0, peakYear: years[0], sigma: 5, base: costs[0] } };
    }

    const minCost = Math.min(...costs);
    const maxCost = Math.max(...costs);
    const peakYear = years[costs.indexOf(maxCost)];
    const sigma = Math.max((years[years.length - 1] - years[0]) / 2, 5);
    const amplitude = maxCost - minCost;

    return {
      predict: (year) => {
        const diff = year - peakYear;
        return Math.max(0, amplitude * Math.exp(-(diff * diff) / (2 * sigma * sigma)) + minCost);
      },
      params: { model: 'bellCurve', amplitude, peakYear, sigma, base: minCost },
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
    return {
      predict: () => avg,
      params:  { model: 'ma3', rate: avg },
    };
  }

  /**
   * Fixed Rate: always returns the user-specified rate.
   */
  function fitFixedRate(rate) {
    return {
      predict: () => rate,
      params:  { model: 'fixedRate', rate },
    };
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────────────────────────────────────
  return {
    /**
     * @param {'logisticLinear'|'careerGamma'} model
     * @param {number[]} years
     * @param {number[]} salaries
     * @param {number}   endYear  — last predicted year
     * @param {object}   [options]
     * @param {number}   [options.alpha=0.8] — prior stiffness for careerGamma
     * @returns {{ predict: (year: number) => number, params: object }}
     */
    fitSalary(model, years, salaries, endYear, options) {
      options = options || {};
      switch (model) {
        case 'careerGamma':
          return fitCareerGamma(years, salaries, endYear, options.alpha, options.expOffset);
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
     * @returns {{ predict: (year: number, salary: number) => number, params: object }}
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
     * @returns {{ predict: (year: number) => number, params: object }}
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
     * Build a salary predictor directly from explicit model parameters.
     * Dispatches on params.model to support both 'logisticLinear' and 'careerGamma'.
     * @param {object} params
     * @returns {(year: number) => number}
     */
    salaryPredictorFromParams(params) {
      if (params.model === 'careerGamma') {
        const { a, k, b, c, minYear, expOffset = 0 } = params;
        return (year) => {
          const x = year - minYear + 1 + expOffset;
          return Math.max(0, gammaEval(x, a, k, b, c));
        };
      }
      // logisticLinear (default)
      const { L, k, m, minYear } = params;
      return (year) => {
        const t = year - minYear;
        return Math.max(0, L * sigmoid(k * (t - m)));
      };
    },

    /**
     * Build a cost predictor from explicit cost model parameters.
     * @param {object} params — cost params object with a 'model' field
     * @returns {(year: number, salary: number) => number}
     */
    costPredictorFromParams(params) {
      if (params.model === 'fixedSaveRate') {
        const { saveRate } = params;
        return (_year, salary) => Math.max(0, salary * (1 - saveRate));
      }
      if (params.model === 'inflationLinear') {
        const { base, annualRate, slope, t0 } = params;
        return (year) => {
          const dt = year - t0;
          return Math.max(0, base * Math.pow(1 + annualRate, dt) + slope * dt);
        };
      }
      if (params.model === 'bellCurve') {
        const { amplitude, peakYear, sigma, base } = params;
        return (year) => {
          const diff = year - peakYear;
          return Math.max(0, amplitude * Math.exp(-(diff * diff) / (2 * sigma * sigma)) + base);
        };
      }
      return () => 0;
    },

    /**
     * Build a return-rate predictor from explicit return model parameters.
     * @param {object} params — return params object with a 'model' field
     * @returns {(year: number) => number}
     */
    returnPredictorFromParams(params) {
      return () => params.rate;
    },
  };
})();
