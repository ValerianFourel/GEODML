#!/usr/bin/env python3
"""Print the top-3 and top-5 summary from a saved direct-axis report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_top_k_summary(report):
    groups = {}
    for row in report["summaries"]:
        key = tuple(row[k] for k in ("model", "method", "engine", "condition"))
        groups.setdefault(key, {})[(row["coordinate_role"], row["metric"])] = row
    if groups and not any(metric.startswith("top5_full_") for rows in groups.values() for _, metric in rows):
        raise ValueError("This report has no full top-5 metrics. Rerun report_axis_ranking_change.py with the updated code.")

    def num(value, *, percent=False, signed=False):
        if value is None:
            return "NA"
        if percent:
            return f"{100 * value:.1f}"
        return f"{value:+.3f}" if signed else f"{value:.3f}"

    lines = ["PROMPT AXIS -> TOP-3 / TOP-5 RANKING DIFFERENCES",
             f"Verified task observations: {report['observations']}",
             f"By model: {report['accounting']['verified_completed_tasks']}",
             "P = Parallel-Expansion; R = Reactive-Snippet-Loop"]
    if not groups:
        lines.append("No eligible cross-prompt comparisons.")
    for condition in ("natural", "shuffled", "ablated"):
        for k in (3, 5):
            lines.extend([f"\n{condition.upper()} | TOP {k} | full lists only",
                          f"{'Model':<8} {'M':<1} {'Engine':<10} {'Pairs':>7} {'KW':>4} {'Short':>6} "
                          f"{'SetΔ%':>6} {'ListΔ%':>7} {'Dist':>6} {'ρ set':>7} {'Within':>7} "
                          f"{'Assigned':>8} {'ρ order':>8} {'OrderN':>7}"])
            for (model, method, engine, cond), metrics in sorted(groups.items()):
                if cond != condition:
                    continue
                prefix = f"top{k}_full_"
                def get(name, role="observed_latent"):
                    return metrics.get((role, prefix + name), {})
                distance, order = get("set_distance"), get("kendall_distance_common")
                rho_key = "rho_axis_gap_vs_ranking_distance"
                lines.append(
                    f"{model:<8} {method[0]:<1} {engine:<10} {distance.get('pairs', 0):>7} "
                    f"{distance.get('keywords', 0):>4} {distance.get('excluded_pairs', 0):>6} "
                    f"{num(get('set_change').get('mean_distance'), percent=True):>6} "
                    f"{num(get('ordered_change').get('mean_distance'), percent=True):>7} "
                    f"{num(distance.get('mean_distance')):>6} "
                    f"{num(distance.get(rho_key), signed=True):>7} "
                    f"{num(distance.get('median_within_keyword_rho'), signed=True):>7} "
                    f"{num(get('set_distance', 'assigned').get(rho_key), signed=True):>8} "
                    f"{num(order.get(rho_key), signed=True):>8} {order.get('pairs', 0):>7}"
                )
    lines.extend(["", "READING THE TABLE",
                  "Pairs/KW: usable prompt pairs and keywords for full top-k set comparisons.",
                  "Short: otherwise eligible nonempty pairs excluded because either ranking has fewer than k URLs.",
                  "SetΔ%: pairs with different top-k URL membership. ListΔ%: membership OR order differs.",
                  "Dist: mean Jaccard set distance, 0 = same members, 1 = disjoint. Order is ignored here.",
                  "ρ set: Spearman correlation between observed axis gap and set distance.",
                  "Within: median set-distance correlation within keywords with a defined correlation.",
                  "Assigned: set-distance correlation for assigned coordinates; equal-coordinate pairs are excluded separately.",
                  "ρ order: axis-gap correlation with Kendall distance among shared top-k URLs.",
                  "OrderN: pairs with at least two common top-k URLs. NA means correlation is undefined.",
                  "Top-3 and top-5 may use different subsets due to short lists. No automatic head-to-head model comparison.",
                  "Pairs reuse prompts. Exact query, candidates and evidence are not matched; no causal claim."])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(format_top_k_summary(json.loads(args.report.read_text())))


if __name__ == "__main__":
    main()
