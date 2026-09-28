"""Keyword-cluster bootstrap: resample whole keywords, keep their prompts, cells and sources."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, cluster: str, statistic: Callable[[list], float | None],
    replicates: int = 2000, seed: int = 20260929, level: float = 0.95,
) -> dict[str, Any]:
    """Percentile interval for ``statistic`` with clusters (e.g. keywords) as the unit.

    Rows are never treated as independent: a replicate draws clusters with
    replacement and includes every row of each drawn cluster. Replicates whose
    statistic is undefined are counted, not dropped silently.
    """

    groups: dict[Any, list] = defaultdict(list)
    for row in rows:
        groups[row[cluster]].append(row)
    keys = sorted(groups, key=str)
    estimate = statistic(list(rows))
    if not keys or estimate is None:
        return {"estimate": estimate, "low": None, "high": None, "clusters": len(keys),
                "replicates": 0, "undefined_replicates": 0}
    rng = random.Random(seed)
    values, undefined = [], 0
    for _ in range(replicates):
        sample = [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]]
        value = statistic(sample)
        if value is None:
            undefined += 1
        else:
            values.append(value)
    values.sort()
    tail = (1 - level) / 2
    pick = lambda q: values[min(len(values) - 1, max(0, int(q * len(values))))] if values else None
    return {"estimate": estimate, "low": pick(tail), "high": pick(1 - tail), "clusters": len(keys),
            "replicates": len(values), "undefined_replicates": undefined}


def mean_of(field: str) -> Callable[[list], float | None]:
    """Mean of a field over rows where it is defined (bools count as 0/1)."""

    def statistic(rows: list) -> float | None:
        values = [float(r[field]) for r in rows if r.get(field) is not None]
        return sum(values) / len(values) if values else None

    return statistic


def paired_difference(first: str, second: str) -> Callable[[list], float | None]:
    """Mean of first - second over rows where both are defined (e.g. A - F)."""

    def statistic(rows: list) -> float | None:
        pairs = [float(r[first]) - float(r[second]) for r in rows
                 if r.get(first) is not None and r.get(second) is not None]
        return sum(pairs) / len(pairs) if pairs else None

    return statistic
