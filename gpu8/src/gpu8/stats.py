"""Statistical analysis with the disease as the unit (plan section 16).

Design decisions the plan fixes and this module enforces:

* the resampling unit is the disease *family*, because diseases in one family
  share ontology structure and reference knowledge; treating them as
  independent would overstate precision;
* training seeds and support-order draws are averaged **inside** a disease
  before any method difference is computed, so repeated runs never inflate n;
* exactly three primary comparisons in two hypothesis families, Holm-corrected;
  everything else is reported as exploratory;
* absolute differences lead; relative change is secondary because a small
  baseline makes a large percentage meaningless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class PairedResult:
    metric: str
    method_a: str
    method_b: str
    n_diseases: int
    n_families: int
    mean_a: float
    mean_b: float
    delta: float
    relative: float
    ci_low: float
    ci_high: float
    permutation_p: float
    wilcoxon_p: float
    frac_benefit: float
    frac_tie: float
    frac_harm: float
    bootstrap_draws: int
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def collapse_repeats(per_disease: pd.DataFrame, metric: str,
                     keys=("method", "disease_id")) -> pd.DataFrame:
    """Average seeds / support draws inside a disease before comparing methods."""
    return (per_disease.groupby(list(keys), as_index=False)[metric]
            .mean().rename(columns={metric: "value"}))


def family_bootstrap(deltas: np.ndarray, families: np.ndarray, draws: int = 10000,
                     seed: int = 20260922, alpha: float = 0.05
                     ) -> tuple[float, float, np.ndarray]:
    """Percentile CI resampling whole families, then re-averaging by disease."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(families)
    by_family = {f: np.flatnonzero(families == f) for f in uniq}
    means = np.empty(draws, dtype=np.float64)
    n_fam = len(uniq)
    for b in range(draws):
        pick = rng.choice(n_fam, size=n_fam, replace=True)
        idx = np.concatenate([by_family[uniq[p]] for p in pick])
        means[b] = deltas[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi, means


def paired_permutation(deltas: np.ndarray, families: np.ndarray, draws: int = 10000,
                       seed: int = 20260922) -> float:
    """Sign-flip permutation at the family level (the exchangeable unit)."""
    rng = np.random.default_rng(seed + 1)
    uniq = np.unique(families)
    by_family = {f: np.flatnonzero(families == f) for f in uniq}
    observed = abs(deltas.mean())
    count = 0
    for _ in range(draws):
        signs = rng.choice([-1.0, 1.0], size=len(uniq))
        flipped = deltas.copy()
        for s, f in zip(signs, uniq):
            flipped[by_family[f]] *= s
        if abs(flipped.mean()) >= observed - 1e-15:
            count += 1
    return (count + 1) / (draws + 1)


def wilcoxon_p(deltas: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon
        nz = deltas[deltas != 0]
        if len(nz) < 5:
            return float("nan")
        return float(wilcoxon(nz).pvalue)
    except Exception:
        return float("nan")


def compare_methods(per_disease: pd.DataFrame, metric: str, method_a: str,
                    method_b: str, family_of: dict[str, str], draws: int = 10000,
                    seed: int = 20260922, note: str = "") -> PairedResult | None:
    """Paired comparison on the diseases both methods actually scored."""
    tab = collapse_repeats(per_disease, metric)
    a = tab[tab["method"] == method_a].set_index("disease_id")["value"]
    b = tab[tab["method"] == method_b].set_index("disease_id")["value"]
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 5:
        return None
    av = a.loc[common].to_numpy(dtype=float)
    bv = b.loc[common].to_numpy(dtype=float)
    keep = ~(np.isnan(av) | np.isnan(bv))
    common = list(np.array(common)[keep])
    av, bv = av[keep], bv[keep]
    if len(common) < 5:
        return None
    deltas = av - bv
    fams = np.array([family_of.get(d, d) for d in common])
    lo, hi, _ = family_bootstrap(deltas, fams, draws=draws, seed=seed)
    p_perm = paired_permutation(deltas, fams, draws=draws, seed=seed)
    return PairedResult(
        metric=metric, method_a=method_a, method_b=method_b,
        n_diseases=len(common), n_families=int(len(np.unique(fams))),
        mean_a=float(av.mean()), mean_b=float(bv.mean()),
        delta=float(deltas.mean()),
        relative=float(deltas.mean() / bv.mean()) if bv.mean() not in (0.0,) else math.nan,
        ci_low=lo, ci_high=hi, permutation_p=p_perm, wilcoxon_p=wilcoxon_p(deltas),
        frac_benefit=float((deltas > 1e-12).mean()),
        frac_tie=float((np.abs(deltas) <= 1e-12).mean()),
        frac_harm=float((deltas < -1e-12).mean()),
        bootstrap_draws=draws, note=note)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment inside one hypothesis family."""
    items = [(k, v) for k, v in pvalues.items() if not (v is None or math.isnan(v))]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adjusted[k] = running
    for k, v in pvalues.items():
        adjusted.setdefault(k, float("nan"))
    return adjusted


def support_gain(per_disease: pd.DataFrame, metric: str, method: str,
                 k_values=(1, 3, 5)) -> pd.DataFrame:
    """Delta_d^(k) = R_d^(k) - R_d^(0) for one method, on the common task set."""
    tab = (per_disease.groupby(["method", "disease_id", "k"], as_index=False)[metric]
           .mean().rename(columns={metric: "value"}))
    sub = tab[tab["method"] == method]
    base = sub[sub["k"] == 0].set_index("disease_id")["value"]
    rows = []
    for k in k_values:
        cur = sub[sub["k"] == k].set_index("disease_id")["value"]
        common = sorted(set(base.index) & set(cur.index))
        for d in common:
            rows.append({"disease_id": d, "k": k, "method": method,
                         "delta": float(cur.loc[d] - base.loc[d]),
                         "value_k": float(cur.loc[d]), "value_0": float(base.loc[d])})
    return pd.DataFrame(rows)


def gamma_comparison(per_disease: pd.DataFrame, metric: str, ours: str, meta: str,
                     family_of: dict[str, str], k_values=(1, 3, 5),
                     draws: int = 10000, seed: int = 20260922) -> pd.DataFrame:
    """Gamma_d^(k) = Delta_d^(k)(ours) - Delta_d^(k)(meta ridge).

    This is the quantity that answers whether the *support mechanism* is better,
    as opposed to merely showing that more support helps.
    """
    go = support_gain(per_disease, metric, ours, k_values)
    gm = support_gain(per_disease, metric, meta, k_values)
    rows = []
    for k in k_values:
        a = go[go["k"] == k].set_index("disease_id")["delta"]
        b = gm[gm["k"] == k].set_index("disease_id")["delta"]
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 5:
            continue
        gamma = a.loc[common].to_numpy() - b.loc[common].to_numpy()
        fams = np.array([family_of.get(d, d) for d in common])
        lo, hi, _ = family_bootstrap(gamma, fams, draws=draws, seed=seed)
        p = paired_permutation(gamma, fams, draws=draws, seed=seed)
        rows.append({"k": k, "metric": metric, "n_diseases": len(common),
                     "n_families": int(len(np.unique(fams))),
                     "mean_delta_ours": float(a.loc[common].mean()),
                     "mean_delta_meta": float(b.loc[common].mean()),
                     "gamma": float(gamma.mean()), "ci_low": lo, "ci_high": hi,
                     "permutation_p": p,
                     "frac_gamma_positive": float((gamma > 0).mean()),
                     "frac_gamma_negative": float((gamma < 0).mean())})
    return pd.DataFrame(rows)


def power_note(sd: float = 0.20, target_delta: float = 0.03, alpha: float = 0.05,
               power: float = 0.80) -> dict:
    """Normal approximation used only for *planning*, never post hoc."""
    z_a, z_b = 1.96, 0.84
    n = ((z_a + z_b) ** 2 * sd ** 2) / (target_delta ** 2)
    return {"assumed_sd": sd, "target_delta": target_delta, "alpha": alpha,
            "power": power, "n_required": float(n),
            "caveat": ("ignores family correlation, multiplicity and non-normal "
                       "metric distributions; planning only")}


def stratified_summary(per_disease: pd.DataFrame, metric: str, method: str,
                       strata: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    sub = per_disease[per_disease["method"] == method]
    for name, mask_ids in strata.items():
        s = sub[sub["disease_id"].isin(set(mask_ids))]
        if s.empty:
            continue
        rows.append({"stratum": name, "n_diseases": int(s["disease_id"].nunique()),
                     "mean": float(s[metric].mean()),
                     "median": float(s[metric].median()),
                     "sd": float(s[metric].std(ddof=1)) if len(s) > 1 else 0.0})
    return pd.DataFrame(rows)
