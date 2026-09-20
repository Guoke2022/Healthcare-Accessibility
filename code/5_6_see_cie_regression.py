# -*- coding: utf-8 -*-
"""Estimate SEE/CIE associations with accessibility and inequality changes.

The public reproduction retains the three-stage HC1 OLS specification used in
the manuscript and writes only manuscript-facing regression tables plus compact
machine-readable coefficient/effect files used by Figure 5.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import BASE_YEAR, END_YEAR, SEE_CIE_PANEL_ROOT, SEE_CIE_REGRESSION_ROOT, CITY_ORDER_4
from utils.extended_analysis import read_csv_robust, zscore_inplace, p_to_star, summary_col_standard_stars

ACC_BASE_COL = f"acc_{BASE_YEAR}"
GINI_BASE_COL = f"gini_{BASE_YEAR}"
THEIL_BASE_COL = f"theil_{BASE_YEAR}"
ATKINSON_BASE_COL = f"atkinson_05_{BASE_YEAR}"
GDP_BASE_COL = f"GDP_{BASE_YEAR}"
RESPOP_BASE_COL = f"ResPop_{BASE_YEAR}"
FISCAL_BASE_COL = f"ln_FiscalRevenue_pc_{BASE_YEAR}"


def _find_interaction(params_index, var_prefix, city, fe_col="city_level_4"):
    matches = [
        name for name in params_index
        if var_prefix in name and f"[T.{city}]" in name and f"C({fe_col}" in name and ":" in name
    ]
    return matches[0] if matches else None


def _city_effect(model, var_prefix, city, ref="Medium/Small City", fe_col="city_level_4"):
    params = model.params
    if var_prefix not in params.index:
        return None

    names = list(params.index)
    contrast = np.zeros((1, len(names)))
    contrast[0, names.index(var_prefix)] = 1.0
    if city != ref:
        interaction = _find_interaction(params.index, var_prefix, city, fe_col)
        if interaction is not None:
            contrast[0, names.index(interaction)] = 1.0

    effect = float((contrast @ params.to_numpy(dtype=float)).item())
    covariance = model.cov_params().to_numpy(dtype=float)
    se = float(np.sqrt((contrast @ covariance @ contrast.T).item()))
    p_value = float(model.wald_test(contrast, scalar=True).pvalue)
    t_critical = float(stats.t.ppf(0.975, df=float(model.df_resid)))
    return {
        "effect": effect,
        "se": se,
        "p": p_value,
        "ci_low": effect - t_critical * se,
        "ci_high": effect + t_critical * se,
        "df_resid": float(model.df_resid),
    }


def _collect_model_coefficients(model, *, outcome, model_family, stage):
    rows = []
    bse = getattr(model, "bse", pd.Series(index=model.params.index, dtype=float))
    pvalues = getattr(model, "pvalues", pd.Series(index=model.params.index, dtype=float))
    for term, coefficient in model.params.items():
        rows.append({
            "outcome": outcome,
            "model_family": model_family,
            "stage": stage,
            "term": term,
            "coef": float(coefficient),
            "se": float(bse.get(term, np.nan)),
            "p": float(pvalues.get(term, np.nan)),
            "n": int(model.nobs),
            "r2": float(model.rsquared),
        })
    return rows


def _collect_city_effects(model, *, outcome, expansion, var_prefix):
    rows = []
    for city in CITY_ORDER_4:
        result = _city_effect(model, var_prefix, city)
        if result is None:
            continue
        rows.append({
            "outcome": outcome,
            "expansion": expansion,
            "city_level": city,
            **result,
            "significance": p_to_star(result["p"]),
            "n": int(model.nobs),
        })
    return rows


def _write_model_table(models, names, output_path) -> None:
    table = summary_col_standard_stars(
        models,
        stars=True,
        model_names=names,
        info_dict={"N": lambda x: f"{int(x.nobs)}", "R2": lambda x: f"{x.rsquared:.3f}"},
        float_format="%0.4f",
    )
    output_path.write_text(table.as_text(), encoding="utf-8")


def main() -> None:
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels is required for the SEE/CIE regression stage") from exc

    SEE_CIE_REGRESSION_ROOT.mkdir(parents=True, exist_ok=True)
    df = read_csv_robust(SEE_CIE_PANEL_ROOT / f"city_{BASE_YEAR}_{END_YEAR}_index.csv")
    if FISCAL_BASE_COL not in df.columns:
        raise KeyError(
            f"Missing {FISCAL_BASE_COL} in the released regression panel. "
            "Rebuild stage 5_5 with baseline fiscal capacity enabled."
        )

    df = df[df["city_level"].isin(CITY_ORDER_4)].copy()
    df["city_level_4"] = pd.Categorical(df["city_level"], categories=CITY_ORDER_4, ordered=True)
    city_fe = 'C(city_level_4, Treatment(reference="Medium/Small City"))'

    for column in ["city_SEE", "city_CIE"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["city_TotalExpansion"] = df["city_SEE"] + df["city_CIE"]

    continuous = [
        "city_SEE", "city_CIE", "city_TotalExpansion", ACC_BASE_COL, GINI_BASE_COL,
        THEIL_BASE_COL, ATKINSON_BASE_COL, GDP_BASE_COL, "GDP_growth", "GDP_growth_pct",
        "Ppo_NetIn", "Ppo_NetIn_rate", RESPOP_BASE_COL, "ResPop_growth",
        "ResPop_growth_rate", "pop_density_mean", FISCAL_BASE_COL,
    ]
    inequality = [
        "gini_delta", "theil_delta", "atkinson_05_delta",
        GINI_BASE_COL, THEIL_BASE_COL, ATKINSON_BASE_COL,
    ]
    df = zscore_inplace(df, continuous)
    df = zscore_inplace(df, inequality)

    controls = (
        f"pop_density_mean + {GDP_BASE_COL} + GDP_growth_pct + {RESPOP_BASE_COL} + "
        f"Ppo_NetIn_rate + ResPop_growth_rate + {FISCAL_BASE_COL}"
    )

    coefficient_rows = []
    effect_rows = []

    acc_columns = [
        "acc_delta", ACC_BASE_COL, "pop_density_mean", "city_SEE", "city_CIE",
        GDP_BASE_COL, "GDP_growth_pct", RESPOP_BASE_COL, "Ppo_NetIn_rate",
        "ResPop_growth_rate", FISCAL_BASE_COL, "city_level_4",
    ]
    df_acc = df[acc_columns].dropna().copy()
    if len(df_acc) < 20:
        raise ValueError(f"Accessibility regression has too few complete cases: N={len(df_acc)}")

    formula_1 = (
        f"acc_delta ~ {ACC_BASE_COL} + pop_density_mean + city_SEE + city_CIE + "
        f"{GDP_BASE_COL} + GDP_growth_pct + {RESPOP_BASE_COL} + Ppo_NetIn_rate + "
        f"ResPop_growth_rate + {FISCAL_BASE_COL}"
    )
    formula_2 = formula_1 + f" + {city_fe}"
    formula_3 = f"acc_delta ~ {ACC_BASE_COL} + {controls} + city_SEE*{city_fe} + city_CIE*{city_fe}"

    acc_m1 = smf.ols(formula_1, data=df_acc).fit(cov_type="HC1")
    acc_m2 = smf.ols(formula_2, data=df_acc).fit(cov_type="HC1")
    acc_m3 = smf.ols(formula_3, data=df_acc).fit(cov_type="HC1")

    coefficient_rows += _collect_model_coefficients(acc_m1, outcome="accessibility", model_family="SEE_CIE", stage=1)
    coefficient_rows += _collect_model_coefficients(acc_m2, outcome="accessibility", model_family="SEE_CIE", stage=2)
    coefficient_rows += _collect_model_coefficients(acc_m3, outcome="accessibility", model_family="SEE_CIE", stage=3)
    effect_rows += _collect_city_effects(acc_m3, outcome="accessibility", expansion="SEE", var_prefix="city_SEE")
    effect_rows += _collect_city_effects(acc_m3, outcome="accessibility", expansion="CIE", var_prefix="city_CIE")

    _write_model_table(
        [acc_m1, acc_m2, acc_m3],
        ["OLS", "OLS + City size FE", "OLS + Expansion × City size"],
        SEE_CIE_REGRESSION_ROOT / "accessibility_models.txt",
    )

    specifications = {
        "gini": {"delta": "gini_delta", "base": GINI_BASE_COL},
        "theil": {"delta": "theil_delta", "base": THEIL_BASE_COL},
        "atkinson_05": {"delta": "atkinson_05_delta", "base": ATKINSON_BASE_COL},
    }

    for outcome, spec in specifications.items():
        y, baseline = spec["delta"], spec["base"]
        columns = [
            y, baseline, "pop_density_mean", GDP_BASE_COL, "GDP_growth_pct", RESPOP_BASE_COL,
            "Ppo_NetIn_rate", "ResPop_growth_rate", FISCAL_BASE_COL, "city_SEE", "city_CIE",
            "city_TotalExpansion", "city_level_4",
        ]
        data = df[columns].dropna().copy()
        if len(data) < 20:
            raise ValueError(f"{outcome} regression has too few complete cases: N={len(data)}")

        see_cie_1 = smf.ols(f"{y} ~ {baseline} + {controls} + city_SEE + city_CIE", data=data).fit(cov_type="HC1")
        see_cie_2 = smf.ols(f"{y} ~ {baseline} + {controls} + city_SEE + city_CIE + {city_fe}", data=data).fit(cov_type="HC1")
        see_cie_3 = smf.ols(f"{y} ~ {baseline} + {controls} + city_SEE*{city_fe} + city_CIE*{city_fe}", data=data).fit(cov_type="HC1")

        total_1 = smf.ols(f"{y} ~ {baseline} + {controls} + city_TotalExpansion", data=data).fit(cov_type="HC1")
        total_2 = smf.ols(f"{y} ~ {baseline} + {controls} + city_TotalExpansion + {city_fe}", data=data).fit(cov_type="HC1")
        total_3 = smf.ols(f"{y} ~ {baseline} + {controls} + city_TotalExpansion*{city_fe}", data=data).fit(cov_type="HC1")

        for stage, model in enumerate([see_cie_1, see_cie_2, see_cie_3], start=1):
            coefficient_rows += _collect_model_coefficients(model, outcome=outcome, model_family="SEE_CIE", stage=stage)
        for stage, model in enumerate([total_1, total_2, total_3], start=1):
            coefficient_rows += _collect_model_coefficients(model, outcome=outcome, model_family="TotalExpansion", stage=stage)

        effect_rows += _collect_city_effects(see_cie_3, outcome=outcome, expansion="SEE", var_prefix="city_SEE")
        effect_rows += _collect_city_effects(see_cie_3, outcome=outcome, expansion="CIE", var_prefix="city_CIE")
        effect_rows += _collect_city_effects(total_3, outcome=outcome, expansion="TotalExpansion", var_prefix="city_TotalExpansion")

        _write_model_table(
            [see_cie_1, see_cie_2, see_cie_3],
            ["OLS", "OLS + City size FE", "OLS + Expansion × City size"],
            SEE_CIE_REGRESSION_ROOT / f"inequality_models_{outcome}.txt",
        )
        _write_model_table(
            [total_1, total_2, total_3],
            ["OLS", "OLS + City size FE", "OLS + Total expansion × City size"],
            SEE_CIE_REGRESSION_ROOT / f"total_expansion_models_{outcome}.txt",
        )

    pd.DataFrame(coefficient_rows).to_csv(
        SEE_CIE_REGRESSION_ROOT / "model_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(effect_rows).to_csv(
        SEE_CIE_REGRESSION_ROOT / "city_size_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )

    names = list(acc_m3.params.index)
    wald_rows = []
    for city in CITY_ORDER_4:
        contrast = np.zeros((1, len(names)))

        def add(term, weight):
            if term in names:
                contrast[0, names.index(term)] += weight

        add("city_SEE", 1.0)
        add("city_CIE", -1.0)
        if city != "Medium/Small City":
            add(f'city_SEE:C(city_level_4, Treatment(reference="Medium/Small City"))[T.{city}]', 1.0)
            add(f'city_CIE:C(city_level_4, Treatment(reference="Medium/Small City"))[T.{city}]', -1.0)
        test = acc_m3.wald_test(contrast, scalar=True)
        wald_rows.append({"city_level": city, "wald_stat": float(test.statistic), "p": float(test.pvalue)})

    pd.DataFrame(wald_rows).to_csv(
        SEE_CIE_REGRESSION_ROOT / "see_vs_cie_wald_accessibility.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"SEE/CIE regression completed | accessibility N={len(df_acc)} -> {SEE_CIE_REGRESSION_ROOT}")


if __name__ == "__main__":
    main()
