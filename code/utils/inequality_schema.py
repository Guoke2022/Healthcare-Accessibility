# -*- coding: utf-8 -*-
"""Single source of truth for formal accessibility-inequality outcomes.

All formal inequality metrics use the same population universe:
finite accessibility >= 0 with finite positive population weights.
"""
from __future__ import annotations

from dataclasses import dataclass

INEQUALITY_SCHEMA_VERSION = "gini_all__theil_t_all__atkinson_0p5_all_v1"
ATKINSON_EPSILON = 0.5


@dataclass(frozen=True)
class InequalityMetric:
    key: str
    column: str
    delta_column: str
    base_prefix: str
    label: str


FORMAL_INEQUALITY_METRICS = (
    InequalityMetric("gini", "pop_gini", "gini_delta", "gini", "Gini"),
    InequalityMetric("theil", "pop_theil", "theil_delta", "theil", "Theil T"),
    InequalityMetric(
        "atkinson_05",
        "pop_atkinson_05",
        "atkinson_05_delta",
        "atkinson_05",
        "Atkinson (ε=0.5)",
    ),
)

INEQUALITY_KEYS = tuple(m.key for m in FORMAL_INEQUALITY_METRICS)
INEQUALITY_COLUMNS = tuple(m.column for m in FORMAL_INEQUALITY_METRICS)
INEQUALITY_DELTA_COLUMNS = tuple(m.delta_column for m in FORMAL_INEQUALITY_METRICS)
METRIC_BY_KEY = {m.key: m for m in FORMAL_INEQUALITY_METRICS}
METRIC_BY_COLUMN = {m.column: m for m in FORMAL_INEQUALITY_METRICS}


def base_column(metric_key: str, year: int) -> str:
    m = METRIC_BY_KEY[metric_key]
    return f"{m.base_prefix}_{int(year)}"
