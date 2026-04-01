/**
 * FIRE Simulator — Simulation Engine
 *
 * Runs a year-by-year simulation tracking:
 *   assets(t+1) = assets(t) × (1 + returnRate) + savings + assetChanges
 * where savings = salary − costs.
 *
 * Two-Phase FIRE model:
 *   capitalIncome  = assets(t) × returnRate   (investment returns this year)
 *   phase1Coverage = capitalIncome / costs    (≥ 1 → Phase 1 FIRE: income covers living costs)
 *   phase2Coverage = capitalIncome / salary   (≥ 1 → Phase 2 FIRE: income replaces full salary)
 */

/* eslint-disable no-unused-vars */
const Simulator = (() => {

  /**
   * @param {Object}   config
   * @param {number}   config.initialAssets — portfolio value at start (before first entry)
   * @param {Object[]} config.yearsData     — array of year entries (historical + predicted)
   * @returns {Object[]} simulation results, one per year
   */
  function run({ initialAssets, yearsData }) {
    let assets = parseFloat(initialAssets) || 0;
    const results = [];

    for (const d of yearsData) {
      const salary      = parseFloat(d.salary)       || 0;
      const costs       = parseFloat(d.costs)        || 0;
      const returnRate  = parseFloat(d.returnRate)   || 0;
      const assetChg    = parseFloat(d.assetChanges) || 0;

      // Capital income = what the portfolio earns this year (start-of-year basis)
      const capitalIncome = assets * returnRate;
      const savings       = salary - costs;
      assets = assets + capitalIncome + savings + assetChg;

      // Phase 1: can capital income cover living costs?
      const phase1Coverage = costs > 0 ? capitalIncome / costs : 0;
      // Phase 2: can capital income replace the full salary?
      const phase2Coverage = salary > 0 ? capitalIncome / salary : 0;

      results.push({
        year:           d.year,
        salary,
        costs,
        savings,
        returnRate,
        capitalIncome,
        assetChanges:   assetChg,
        assets,
        phase1Coverage,
        phase2Coverage,
        isPhase1FIRE:   phase1Coverage >= 1,
        isPhase2FIRE:   phase2Coverage >= 1,
        isHistorical:   d.isHistorical,
      });
    }

    return results;
  }

  return { run };
})();
