/**
 * FIRE Simulator — Simulation Engine
 *
 * Runs a year-by-year simulation tracking:
 *   assets(t+1) = assets(t) × (1 + returnRate) + savings + assetChanges
 * where savings = salary − costs.
 *
 * Also computes the FIRE coverage ratio:
 *   coverage = assets × safeWithdrawalRate / costs
 * FIRE is reached when coverage ≥ 1.
 */

/* eslint-disable no-unused-vars */
const Simulator = (() => {

  /**
   * @param {Object} config
   * @param {number}   config.initialAssets   — portfolio value at start (before first entry)
   * @param {Object[]} config.yearsData        — array of year entries (historical + predicted)
   * @param {number}   config.swr              — safe withdrawal rate as decimal (e.g. 0.04)
   * @returns {Object[]} simulation results, one per year
   */
  function run({ initialAssets, yearsData, swr }) {
    let assets = parseFloat(initialAssets) || 0;
    const results = [];

    for (const d of yearsData) {
      const salary      = parseFloat(d.salary)       || 0;
      const costs       = parseFloat(d.costs)        || 0;
      const returnRate  = parseFloat(d.returnRate)   || 0;
      const assetChg    = parseFloat(d.assetChanges) || 0; // optional extra

      const assetGrowth = assets * returnRate;
      const savings     = salary - costs;
      assets = assets + assetGrowth + savings + assetChg;

      const coverageRatio = costs > 0 ? (assets * swr) / costs : 0;

      results.push({
        year:          d.year,
        salary,
        costs,
        savings,
        returnRate,
        assetGrowth,
        assetChanges:  assetChg,
        assets,
        coverageRatio,
        isFIRE:        coverageRatio >= 1,
        isHistorical:  d.isHistorical,
      });
    }

    return results;
  }

  return { run };
})();
