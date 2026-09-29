#!/usr/bin/env python
"""
RTOGuard AI - Step 7: Threshold analysis (OLD and NEW customer models, analysed separately)

Pipeline:  raw RTO probability -> threshold sweep -> review-capacity scenarios
           -> illustrative cost model -> candidate thresholds (NOT final risk bands)

Rules honoured: no retraining, no feature changes, no split changes, no recalibration,
raw predict_proba(X_test)[:, 1] only, no previous outputs overwritten.

Usage:
    python step7_threshold_analysis.py                 # real run (edit CONFIG first)
    python step7_threshold_analysis.py --demo --outdir /tmp/demo_out   # synthetic smoke test only
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================================
# CONFIG - EDIT THIS SECTION
# =====================================================================================
OUTPUT_DIR = "step7_threshold_analysis"

# ---- Where the test probabilities come from --------------------------------------
# Option 1 (preferred): existing test-probability CSVs from the earlier steps.
# Option 2: recreate the identical test set and load the saved .joblib model (no retraining).
MODELS = {
    "OLD": {
     "prob_file": "old_test_probabilities.csv",
"y_col": "actual_rto",
"p_col": "predicted_probability",                              # <-- edit (raw probability)
        "recreate": {                                         # used only if prob_file is missing
            "data_path": "PATH/TO/old_cleaned.csv",           # <-- edit
            "target_col": "TARGET_COLUMN",                    # <-- edit
            "feature_cols": [],                               # <-- SAME feature list as training
            "forbidden_cols": [],                             # <-- SAME forbidden-column rules
            "model_path": "PATH/TO/old_model.joblib",         # <-- edit
            "preprocessor_path": None,                        # only if preprocessing is a separate object
        },
    },
    "NEW": {
       "prob_file": "new_test_probabilities.csv",
"y_col": "actual_rto",
"p_col": "predicted_probability",
        "recreate": {
            "data_path": "PATH/TO/new_cleaned.csv",
            "target_col": "TARGET_COLUMN",
            "feature_cols": [],
            "forbidden_cols": [],
            "model_path": "PATH/TO/new_model.joblib",
            "preprocessor_path": None,
        },
    },
}
TEST_SIZE, RANDOM_STATE = 0.20, 42          # must match the earlier split

# ---- Context statistics from earlier steps (documentation only, not recomputed) ---
MODEL_CONTEXT = {
    "OLD": {"roc_auc": 0.7724, "top5_pred": 0.8003, "top5_obs": 0.7667,
            "bot5_pred": 0.0504, "bot5_obs": 0.0417},
    "NEW": {"roc_auc": 0.6698, "top5_pred": 0.7420, "top5_obs": 0.8250,
            "bot5_pred": 0.1538, "bot5_obs": 0.1750},
}

# ---- COST MODEL: ILLUSTRATIVE - NOT REAL BUSINESS DATA -----------------------------
# Absolute values are placeholders in arbitrary currency units; only RATIOS matter.
RTO_COST = 100.0                             # ILLUSTRATIVE cost of one RTO (shipping both ways, handling, lost stock time)
INTERVENTION_SUCCESS_RATE = 1.0              # ILLUSTRATIVE: share of flagged true RTOs that intervention prevents.
                                             # 1.0 = best case; try 0.3-0.7 for realism.
INTERVENTION_COST_RATIOS = {                 # ILLUSTRATIVE: intervention cost as fraction of RTO cost
    "A_low":    0.10,
    "B_medium": 0.25,
    "C_high":   0.50,
}
RTO_COST_SENSITIVITY = [50.0, 100.0, 200.0, 400.0]   # ILLUSTRATIVE: vary RTO cost ...
FIXED_INTERVENTION_COST = 25.0                        # ... while intervention cost is held fixed
REVIEW_CAPACITY = 0.15                       # ILLUSTRATIVE optional cap: max share of orders reviewable. None = off.
CAPACITY_SCENARIOS = [0.05, 0.10, 0.15, 0.20]

THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]  # superset of required 0.10-0.90 grid
KEY_THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 0.90]                    # quoted in the report

DISCLAIMER = "ILLUSTRATIVE - NOT REAL BUSINESS DATA"
CANDIDATE_NOTE = ("These are threshold candidates based on stated assumptions, "
                  "not universal optimal values.")


# =====================================================================================
# LOADING
# =====================================================================================
def load_test_probs(name, cfg):
    """Return (y_true, raw_probability) as numpy arrays for one model."""
    if os.path.exists(cfg["prob_file"]):
        df = pd.read_csv(cfg["prob_file"])
        return df[cfg["y_col"]].to_numpy().astype(int), df[cfg["p_col"]].to_numpy().astype(float)

    r = cfg["recreate"]
    if not os.path.exists(r["data_path"]) or not r["feature_cols"]:
        sys.exit(f"[{name}] No probability file at '{cfg['prob_file']}' and 'recreate' is not configured. "
                 f"Edit CONFIG.")
    import joblib
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(r["data_path"])
    bad = set(r["feature_cols"]) & set(r["forbidden_cols"])
    assert not bad, f"[{name}] forbidden columns present in feature list: {bad}"
    X, y = df[r["feature_cols"]], df[r["target_col"]].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                            random_state=RANDOM_STATE, stratify=y)
    if r.get("preprocessor_path"):
        X_test = joblib.load(r["preprocessor_path"]).transform(X_test)
    model = joblib.load(r["model_path"])                     # no retraining
    return y_test.to_numpy(), model.predict_proba(X_test)[:, 1]


def synthetic_probs(seed, noise):
    """DEMO ONLY - synthetic data to smoke-test the pipeline. Not real results."""
    rng = np.random.default_rng(seed)
    n = 6000
    latent = rng.normal(-0.5, 1.4, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(int)
    p = 1 / (1 + np.exp(-(latent + rng.normal(0, noise, n)) * 0.8))
    return y, p


# =====================================================================================
# METRICS
# =====================================================================================
def _div(a, b):
    return a / b if b else np.nan


def mask_metrics(y, mask):
    n = len(y)
    tp = int(((mask) & (y == 1)).sum()); fp = int(((mask) & (y == 0)).sum())
    fn = int(((~mask) & (y == 1)).sum()); tn = int(((~mask) & (y == 0)).sum())
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else np.nan
    return {
        "predicted_positive_count": tp + fp, "percentage_flagged": 100 * (tp + fp) / n,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": prec, "recall": rec, "F1": f1,
        "false_positive_rate": _div(fp, fp + tn), "false_negative_rate": _div(fn, fn + tp),
        "rto_rate_among_flagged": prec, "rto_rate_among_unflagged": _div(fn, fn + tn),
        "rtos_caught": tp, "rtos_missed": fn, "delivered_unnecessarily_flagged": fp,
    }


def cost_metrics(y, p, mask, R, I, eff):
    """
    Realised cost (uses actual outcomes):
        intervention cost  = flagged * I
        RTO cost           = (missed RTOs + caught RTOs the intervention fails to prevent) * R
    Model-expected cost (uses probabilities only - what the business would 'expect' at deploy time):
        flagged * I  +  sum(p over unflagged) * R  +  sum(p over flagged) * (1 - eff) * R
    Baseline = no intervention at all = (all actual RTOs) * R.
    """
    n, flagged = len(y), int(mask.sum())
    tp, fn = int((mask & (y == 1)).sum()), int((~mask & (y == 1)).sum())
    realised = flagged * I + (fn + tp * (1 - eff)) * R
    expected = flagged * I + p[~mask].sum() * R + p[mask].sum() * (1 - eff) * R
    baseline = (tp + fn) * R
    return {"intervention_cost_total": flagged * I,
            "residual_rto_cost_total": (fn + tp * (1 - eff)) * R,
            "realised_total_cost": realised, "model_expected_total_cost": expected,
            "baseline_no_intervention_cost": baseline,
            "net_saving_vs_baseline": baseline - realised,
            "realised_cost_per_order": realised / n, "expected_cost_per_order": expected / n}


def sweep(y, p, thresholds):
    rows = []
    for t in thresholds:
        r = {"threshold": t, "risk_score_equivalent": round(t * 100)}
        r.update(mask_metrics(y, p >= t))
        rows.append(r)
    return pd.DataFrame(rows)


def capacity_table(y, p, caps):
    order = np.argsort(-p, kind="stable")
    rows = []
    for c in caps:
        k = int(round(c * len(y)))
        mask = np.zeros(len(y), bool); mask[order[:k]] = True   # rank-based -> exactly k orders
        m = mask_metrics(y, mask)
        rows.append({"review_capacity_pct": 100 * c, "probability_cutoff": float(p[order[k - 1]]),
                     "risk_score_cutoff": round(float(p[order[k - 1]]) * 100, 1),
                     "orders_reviewed": m["predicted_positive_count"],
                     "actual_rtos_caught": m["TP"], "recall_all_rtos": m["recall"],
                     "precision_reviewed": m["precision"],
                     "rto_rate_reviewed": m["rto_rate_among_flagged"],
                     "rto_rate_not_reviewed": m["rto_rate_among_unflagged"],
                     "delivered_unnecessarily_reviewed": m["FP"],
                     "note": "scenario only - not claimed optimal"})
    return pd.DataFrame(rows), order


def scenarios():
    """(sensitivity_type, scenario_name, R, I) - ALL ILLUSTRATIVE."""
    out = [("intervention_ratio", f"{k} ({int(v * 100)}% of RTO cost)", RTO_COST, RTO_COST * v)
           for k, v in INTERVENTION_COST_RATIOS.items()]
    out += [("rto_cost", f"RTO_cost={r:g} (interv={FIXED_INTERVENTION_COST:g})", r, FIXED_INTERVENTION_COST)
            for r in RTO_COST_SENSITIVITY]
    return out


def cost_sensitivity(y, p, thresholds, cap):
    rows = []
    for stype, sname, R, I in scenarios():
        for t in thresholds:
            mask = p >= t
            r = {"sensitivity_type": stype, "scenario": sname, "assumption_label": DISCLAIMER,
                 "RTO_cost": R, "intervention_cost": I, "intervention_to_rto_ratio": I / R,
                 "intervention_success_rate": INTERVENTION_SUCCESS_RATE,
                 "analytic_break_even_probability": I / (INTERVENTION_SUCCESS_RATE * R),
                 "threshold": t, "percentage_flagged": 100 * mask.mean(),
                 "within_review_capacity": (mask.mean() <= cap) if cap else np.nan}
            r.update(cost_metrics(y, p, mask, R, I, INTERVENTION_SUCCESS_RATE))
            rows.append(r)
    df = pd.DataFrame(rows)
    df["is_min_cost_unconstrained"] = df.groupby("scenario")["realised_total_cost"].transform(
        lambda s: s == s.min())
    return df


# =====================================================================================
# CANDIDATES
# =====================================================================================
def build_candidates(model, y, p, sweep_df, cs_df, cap):
    rows = []

    def add(scn, rule, t, mask, R, I):
        m, c = mask_metrics(y, mask), cost_metrics(y, p, mask, R, I, INTERVENTION_SUCCESS_RATE)
        rows.append({"model": model, "scenario": scn, "rule": rule, "candidate_threshold": round(t, 4),
                     "pct_flagged": m["percentage_flagged"], "precision": m["precision"],
                     "recall": m["recall"], "expected_cost_per_order": c["expected_cost_per_order"],
                     "realised_cost_per_order": c["realised_cost_per_order"],
                     "net_saving_vs_baseline": c["net_saving_vs_baseline"],
                     "review_load_orders": m["predicted_positive_count"],
                     "assumption_label": DISCLAIMER, "status": "CANDIDATE - not final"})

    for stype, sname, R, I in scenarios():
        sub = cs_df[cs_df.scenario == sname]
        best = sub.loc[sub.realised_total_cost.idxmin()]
        add(sname, "min realised cost (unconstrained, grid)", best.threshold, p >= best.threshold, R, I)
        be = I / (INTERVENTION_SUCCESS_RATE * R)
        add(sname, "analytic break-even p = I/(eff*R)", be, p >= be, R, I)
        if cap:
            ok = sub[sub.within_review_capacity == True]
            if len(ok):
                b = ok.loc[ok.realised_total_cost.idxmin()]
                add(sname, f"min realised cost within {cap:.0%} review capacity", b.threshold,
                    p >= b.threshold, R, I)
            else:
                rows.append({"model": model, "scenario": sname,
                             "rule": f"within {cap:.0%} capacity", "candidate_threshold": np.nan,
                             "status": "no grid threshold fits capacity"})
    return pd.DataFrame(rows)


# =====================================================================================
# PLOTS
# =====================================================================================
def plot_tradeoffs(name, sw, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    panels = [("precision", "Precision"), ("recall", "Recall"), ("F1", "F1"),
              ("percentage_flagged", "% of orders flagged")]
    for a, (col, title) in zip(ax.ravel(), panels):
        a.plot(sw.threshold, sw[col], marker="o")
        a.set_title(f"{name}: threshold vs {title}"); a.set_xlabel("probability threshold")
        a.set_ylabel(title); a.grid(alpha=.3)
    fig.suptitle(f"{name} model - precision/recall trade-off (raw probabilities, test set)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_cost(name, cs, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = cs[cs.sensitivity_type == "intervention_ratio"]
    for i, (sn, g) in enumerate(sub.groupby("scenario")):
        col = f"C{i}"
        ax.plot(g.threshold, g.realised_cost_per_order, marker="o", color=col, label=f"{sn} - realised")
        ax.plot(g.threshold, g.expected_cost_per_order, ls="--", color=col, alpha=.6, label=f"{sn} - model-expected")
        b = g.loc[g.realised_cost_per_order.idxmin()]
        ax.scatter([b.threshold], [b.realised_cost_per_order], s=140, marker="*", color=col, zorder=5)
    base = sub.baseline_no_intervention_cost.iloc[0] / (sub.realised_total_cost.iloc[0] / sub.realised_cost_per_order.iloc[0])
    ax.axhline(base, color="k", ls=":", label="no intervention baseline")
    ax.set_xlabel("probability threshold"); ax.set_ylabel("cost per order (illustrative units)")
    ax.set_title(f"{name}: threshold vs expected cost\n{DISCLAIMER}  (stars = grid minimum)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# =====================================================================================
# REPORT
# =====================================================================================
def fmt(v, pct=False, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.{d}f}"


def at(sw, t, col):
    r = sw[np.isclose(sw.threshold, t)]
    return r[col].iloc[0] if len(r) else np.nan


def build_report(data, summary, cap):
    L = []
    w = L.append
    w("=" * 92); w("RTOGuard AI - STEP 7: THRESHOLD ANALYSIS REPORT"); w("=" * 92)
    w(CANDIDATE_NOTE.upper())
    w(f"ALL COST FIGURES ARE {DISCLAIMER}. Only cost RATIOS are meaningful, and only as sensitivity inputs.")
    w("Raw probabilities used for both models (no recalibration). No retraining, features or split changed.")
    w(f"Intervention success rate assumed: {INTERVENTION_SUCCESS_RATE:.0%} (ILLUSTRATIVE). "
      f"Review capacity cap for constrained candidates: {'off' if not cap else f'{cap:.0%} (ILLUSTRATIVE)'}\n")

    w("METHOD")
    w("  risk_score = round(raw_probability * 100)   (e.g. 0.73 -> 73). No Low/Medium/High bands are defined here.")
    w("  Break-even logic: intervene on an order when  eff * p * RTO_cost > intervention_cost,")
    w("  i.e. p > intervention_cost / (eff * RTO_cost). Example: intervention = 25% of RTO cost, eff = 1 -> p > 0.25.")
    w("  This simple rule is NOT used alone: review capacity, false positives and grid results are also shown.")
    w("  Realised cost uses actual test outcomes; model-expected cost uses predicted probabilities only.\n")

    for name in data:
        d = data[name]; sw, cap_df = d["sweep"], d["capacity"]
        ctx = MODEL_CONTEXT.get(name, {})
        w("-" * 92); w(f"{name} MODEL  (test orders = {len(d['y'])}, actual RTO rate = {d['y'].mean():.1%}, "
                        f"ROC-AUC from earlier step = {ctx.get('roc_auc', 'n/a')})"); w("-" * 92)
        cols = ["threshold", "risk_score_equivalent", "predicted_positive_count", "percentage_flagged",
                "TP", "FP", "TN", "FN", "precision", "recall", "F1", "false_positive_rate"]
        w(sw[cols].round(3).to_string(index=False)); w("")
        w("Review-capacity scenarios (scenario analysis only - no level is claimed optimal):")
        w(cap_df.drop(columns=["note"]).round(3).to_string(index=False)); w("")

    w("=" * 92); w("ANSWERS TO THE REQUIRED QUESTIONS"); w("=" * 92)
    for name in data:
        sw = data[name]["sweep"]
        w(f"\n[{name}]")
        w(" 1. Precision vs threshold: " + "; ".join(
            f"t={t:.2f}: {fmt(at(sw, t, 'precision'), True)}" for t in KEY_THRESHOLDS)
          + ". (n/a = nothing flagged at that cutoff.)")
        w(" 2. Recall vs threshold:    " + "; ".join(
            f"t={t:.2f}: {fmt(at(sw, t, 'recall'), True)}" for t in KEY_THRESHOLDS)
          + ". Recall falls as the cutoff rises; precision usually rises but gets noisy where few orders remain.")
        w(" 3. Orders flagged:         " + "; ".join(
            f"t={t:.2f}: {int(at(sw, t, 'predicted_positive_count'))} ({at(sw, t, 'percentage_flagged'):.1f}%)"
            for t in KEY_THRESHOLDS))
        w(" 4. Actual RTOs caught:     " + "; ".join(
            f"t={t:.2f}: {int(at(sw, t, 'rtos_caught'))} of {int(at(sw, t, 'rtos_caught') + at(sw, t, 'rtos_missed'))}"
            for t in KEY_THRESHOLDS))
        w(" 5. Delivered orders unnecessarily flagged: " + "; ".join(
            f"t={t:.2f}: {int(at(sw, t, 'delivered_unnecessarily_flagged'))}" for t in KEY_THRESHOLDS))
        cd = data[name]["capacity"]
        w(" 6. Review capacity: capacity fixes the cutoff regardless of cost: " + "; ".join(
            f"{r.review_capacity_pct:.0f}% -> p>={r.probability_cutoff:.2f} (recall {r.recall_all_rtos:.0%}, "
            f"precision {r.precision_reviewed:.0%})" for r in cd.itertuples())
          + ". Where the cost-minimising cutoff flags more than capacity allows, capacity becomes the binding constraint.")
        s = summary[summary.model == name]
        for sn_group, label, q in (("intervention_ratio", "intervention cost", 7), ("rto_cost", "RTO cost", 8)):
            sub = s[s.rule.str.startswith("min realised cost (unconstrained")]
            sub = sub[sub.scenario.isin([x[1] for x in scenarios() if x[0] == sn_group])]
            w(f" {q}. Effect of {label} (grid-minimum realised cost, {DISCLAIMER}): " + "; ".join(
                f"[{r.scenario}] t={r.candidate_threshold:.2f}, {r.pct_flagged:.0f}% flagged"
                for r in sub.itertuples()))
        w("    Higher intervention cost (or lower RTO cost) raises the break-even probability -> higher cutoff, "
          "fewer orders flagged. Only the ratio intervention/RTO matters.")

    w("\n 9. OLD vs NEW")
    for name in data:
        c, sw = MODEL_CONTEXT.get(name, {}), data[name]["sweep"]
        y, p = data[name]["y"], data[name]["p"]
        k = max(1, int(round(0.05 * len(y)))); o = np.argsort(-p)
        w(f"    {name}: ROC-AUC {c.get('roc_auc', 'n/a')}; earlier top-5% pred {c.get('top5_pred', 0):.1%} / obs "
          f"{c.get('top5_obs', 0):.1%}; bottom-5% pred {c.get('bot5_pred', 0):.1%} / obs {c.get('bot5_obs', 0):.1%}. "
          f"Recomputed here: top-5% obs {y[o[:k]].mean():.1%}, bottom-5% obs {y[o[-k:]].mean():.1%}.")
    w("    NEW has weaker discrimination, especially in the low-risk region (bottom-5% observed RTO ~17.5% vs ~4.2% for "
      "OLD), so no order can be treated as truly 'safe' under NEW. OLD thresholds must NOT be reused for NEW; "
      "the two models were analysed separately above.")

    w("\n10. CANDIDATE THRESHOLDS FOR BUSINESS DISCUSSION")
    w(f"    {CANDIDATE_NOTE}\n")
    w("| Model | Scenario | Rule | Candidate threshold | % flagged | Precision | Recall | Expected cost/order | Review load |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|")
    show = summary[summary.scenario.str.startswith(tuple(INTERVENTION_COST_RATIOS))]
    for r in show.itertuples():
        if np.isnan(r.candidate_threshold):
            w(f"| {r.model} | {r.scenario} | {r.rule} | none | - | - | - | - | - |"); continue
        w(f"| {r.model} | {r.scenario} | {r.rule} | {r.candidate_threshold:.2f} | {r.pct_flagged:.1f}% | "
          f"{fmt(r.precision, True)} | {fmt(r.recall, True)} | {r.expected_cost_per_order:.2f} | {r.review_load_orders} |")
    w("\n    Exploratory action-level structure (Part 8; NOT final bands): use the low-cost (A) and high-cost (C) "
      "grid-minimum cutoffs as edges of a middle 'monitoring / soft verification' zone.")
    for name in data:
        s = summary[(summary.model == name) & summary.rule.str.startswith("min realised cost (unconstrained")]
        ta = s[s.scenario.str.startswith("A_")].candidate_threshold.iloc[0]
        tc = s[s.scenario.str.startswith("C_")].candidate_threshold.iloc[0]
        lo, hi = min(ta, tc), max(ta, tc); y, p = data[name]["y"], data[name]["p"]
        zones = [("Normal processing", p < lo), ("Monitoring / soft verification", (p >= lo) & (p < hi)),
                 ("Strong verification / intervention", p >= hi)]
        w(f"    {name}: edges {lo:.2f} / {hi:.2f}")
        for zn, m in zones:
            w(f"       {zn:<36} {m.sum():>6} orders ({m.mean():5.1%}), observed RTO rate {fmt(y[m].mean() if m.any() else np.nan, True)}")
    w("\nNEXT: Step 8 - finalise risk bands + business action rules, only after real costs, capacity and "
      "intervention effectiveness are confirmed.")
    w(f"\n{CANDIDATE_NOTE}")
    return "\n".join(L)


# =====================================================================================
# MAIN
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data smoke test (NOT real results)")
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    def out(fn):
        pth = os.path.join(a.outdir, fn)
        if os.path.exists(pth):
            sys.exit(f"Refusing to overwrite existing file: {pth}")
        return pth

    data, summaries = {}, []
    for name, cfg in MODELS.items():
        y, p = (synthetic_probs(1 if name == "OLD" else 2, 0.8 if name == "OLD" else 2.0)
                if a.demo else load_test_probs(name, cfg))
        assert set(np.unique(y)) <= {0, 1} and (p >= 0).all() and (p <= 1).all()
        low = name.lower()
        sw = sweep(y, p, THRESHOLDS)
        cap_df, _ = capacity_table(y, p, CAPACITY_SCENARIOS)
        cs = cost_sensitivity(y, p, THRESHOLDS, REVIEW_CAPACITY)
        sw.to_csv(out(f"{low}_threshold_analysis.csv"), index=False)
        cap_df.to_csv(out(f"{low}_review_capacity.csv"), index=False)
        cs.to_csv(out(f"{low}_cost_sensitivity.csv"), index=False)
        plot_tradeoffs(name, sw, out(f"{low}_threshold_tradeoffs.png"))
        plot_cost(name, cs, out(f"{low}_expected_cost.png"))
        summaries.append(build_candidates(name, y, p, sw, cs, REVIEW_CAPACITY))
        data[name] = {"y": y, "p": p, "sweep": sw, "capacity": cap_df}

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out("threshold_analysis_summary.csv"), index=False)
    report = build_report(data, summary, REVIEW_CAPACITY)
    if a.demo:
        report = "*** DEMO RUN ON SYNTHETIC DATA - NOT REAL RESULTS ***\n\n" + report
    with open(out("step7_threshold_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Done. Outputs in {a.outdir}/")


if __name__ == "__main__":
    main()
