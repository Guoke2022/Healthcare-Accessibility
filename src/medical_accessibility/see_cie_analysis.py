from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CITY_ORDER_4 = ["Medium/Small City", "Large City", "Super City", "Mega City"]
METRIC_ORDER = ["gini", "theil", "mld"]
METRIC_COLORS = {
    "gini": "#92b4b8",
    "theil": "#f3d7a3",
    "mld": "#cfa0a2",
}
CITY_LABELS = ["Medium & Small Cities", "Large Cities", "Super Cities", "Mega Cities"]


def load_city_panel(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8")
    if "city_level" not in df.columns:
        raise KeyError("Input data must contain a 'city_level' column.")

    df = df[df["city_level"].isin(CITY_ORDER_4)].copy()
    df["city_level_4"] = pd.Categorical(df["city_level"], categories=CITY_ORDER_4, ordered=True)

    for column in ["city_SEE", "city_CIE"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["city_TotalExpansion"] = df["city_SEE"] + df["city_CIE"]
    return df


def zscore_inplace(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    columns = [column for column in columns if column in df.columns]
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    for column in columns:
        series = df[column]
        mean = series.mean(skipna=True)
        std = series.std(skipna=True)
        if pd.isna(std) or std == 0:
            continue
        df[column] = (series - mean) / std
    return df


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cont_vars = [
        "city_SEE",
        "city_CIE",
        "city_TotalExpansion",
        "acc_2014",
        "gini_2014",
        "theil_2014",
        "MLD_2014",
        "GDP_2014",
        "GDP_growth",
        "GDP_growth_pct",
        "Ppo_NetIn",
        "Ppo_NetIn_rate",
        "ResPop_2014",
        "ResPop_growth",
        "ResPop_growth_rate",
        "pop_density_mean",
    ]
    fair_vars = [
        "gini_delta",
        "theil_delta",
        "MLD_delta",
        "gini_2014",
        "theil_2014",
        "MLD_2014",
    ]
    df = zscore_inplace(df.copy(), cont_vars)
    df = zscore_inplace(df, fair_vars)
    return df


def compute_vif_table(df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_vars = [
        "acc_2014",
        "pop_density_mean",
        "city_SEE",
        "city_CIE",
        "GDP_2014",
        "GDP_growth_pct",
        "ResPop_2014",
        "Ppo_NetIn_rate",
        "ResPop_growth_rate",
    ]
    vif_vars = [column for column in vif_vars if column in df.columns]
    x_vif = df[vif_vars].dropna().copy()
    x_vif["intercept"] = 1.0
    vif_df = pd.DataFrame(
        {
            "variable": x_vif.columns,
            "VIF": [variance_inflation_factor(x_vif.values, i) for i in range(x_vif.shape[1])],
        }
    )
    return vif_df[vif_df["variable"] != "intercept"].copy()


def model_summary_text(models, model_names: list[str]) -> str:
    from statsmodels.iolib.summary2 import summary_col

    table = summary_col(
        models,
        stars=True,
        model_names=model_names,
        info_dict={"N": lambda x: f"{int(x.nobs)}", "R2": lambda x: f"{x.rsquared:.3f}"},
        float_format="%0.4f",
    )
    return table.as_text()


def run_see_cie_models(df: pd.DataFrame, output_dir: Path) -> None:
    import statsmodels.formula.api as smf

    output_dir.mkdir(parents=True, exist_ok=True)
    city_fe_4 = 'C(city_level_4, Treatment(reference="Medium/Small City"))'

    df = prepare_model_dataframe(df)
    compute_vif_table(df).to_csv(output_dir / "vif_acc.csv", index=False, encoding="utf-8-sig")

    acc_vars_for_sample = [
        "acc_delta",
        "acc_2014",
        "pop_density_mean",
        "city_SEE",
        "city_CIE",
        "GDP_2014",
        "GDP_growth_pct",
        "ResPop_2014",
        "Ppo_NetIn_rate",
        "ResPop_growth_rate",
        "city_level_4",
    ]
    acc_vars_for_sample = [column for column in acc_vars_for_sample if column in df.columns]
    df_acc = df[acc_vars_for_sample].dropna().copy()

    formula_acc_1 = (
        "acc_delta ~ acc_2014 + pop_density_mean + city_SEE + city_CIE + "
        "GDP_2014 + GDP_growth_pct + ResPop_2014 + Ppo_NetIn_rate + ResPop_growth_rate"
    )
    formula_acc_2 = (
        "acc_delta ~ acc_2014 + pop_density_mean + city_SEE + city_CIE + "
        "GDP_2014 + GDP_growth_pct + ResPop_2014 + Ppo_NetIn_rate + ResPop_growth_rate + "
        f"{city_fe_4}"
    )
    formula_acc_3 = (
        "acc_delta ~ acc_2014 + pop_density_mean + GDP_2014 + GDP_growth_pct + "
        "ResPop_2014 + Ppo_NetIn_rate + ResPop_growth_rate + "
        f"city_SEE*{city_fe_4} + city_CIE*{city_fe_4}"
    )

    model_acc_1 = smf.ols(formula=formula_acc_1, data=df_acc).fit(cov_type="HC1")
    model_acc_2 = smf.ols(formula=formula_acc_2, data=df_acc).fit(cov_type="HC1")
    model_acc_3 = smf.ols(formula=formula_acc_3, data=df_acc).fit(cov_type="HC1")

    with open(output_dir / "table_acc_models.txt", "w", encoding="utf-8") as handle:
        handle.write(
            model_summary_text(
                [model_acc_1, model_acc_2, model_acc_3],
                ["OLS", "OLS + City size FE", "OLS + Expansion x City size"],
            )
        )

    model_acc_3.params.rename("coef").to_csv(output_dir / "model_acc3_params.csv", encoding="utf-8-sig")
    model_acc_3.cov_params().to_csv(output_dir / "model_acc3_cov.csv", encoding="utf-8-sig")
    pd.Series({"df_resid": model_acc_3.df_resid}).to_csv(
        output_dir / "model_acc3_df.csv",
        encoding="utf-8-sig",
    )

    fairness_specs = {
        "gini": {"delta": "gini_delta", "base": "gini_2014"},
        "theil": {"delta": "theil_delta", "base": "theil_2014"},
        "mld": {"delta": "MLD_delta", "base": "MLD_2014"},
    }
    controls = (
        "pop_density_mean + GDP_2014 + GDP_growth_pct + "
        "ResPop_2014 + Ppo_NetIn_rate + ResPop_growth_rate"
    )

    all_results = {}
    all_results_total = {}
    for metric, spec in fairness_specs.items():
        outcome = spec["delta"]
        baseline = spec["base"]
        vars_for_sample = [
            outcome,
            baseline,
            "pop_density_mean",
            "GDP_2014",
            "GDP_growth_pct",
            "ResPop_2014",
            "Ppo_NetIn_rate",
            "ResPop_growth_rate",
            "city_SEE",
            "city_CIE",
            "city_TotalExpansion",
            "city_level_4",
        ]
        vars_for_sample = [column for column in vars_for_sample if column in df.columns]
        df_fair = df[vars_for_sample].dropna().copy()

        see_formula_1 = f"{outcome} ~ {baseline} + {controls} + city_SEE + city_CIE"
        see_formula_2 = f"{outcome} ~ {baseline} + {controls} + city_SEE + city_CIE + {city_fe_4}"
        see_formula_3 = f"{outcome} ~ {baseline} + {controls} + city_SEE*{city_fe_4} + city_CIE*{city_fe_4}"

        total_formula_1 = f"{outcome} ~ {baseline} + {controls} + city_TotalExpansion"
        total_formula_2 = f"{outcome} ~ {baseline} + {controls} + city_TotalExpansion + {city_fe_4}"
        total_formula_3 = f"{outcome} ~ {baseline} + {controls} + city_TotalExpansion*{city_fe_4}"

        see_models = [
            smf.ols(see_formula_1, data=df_fair).fit(cov_type="HC1"),
            smf.ols(see_formula_2, data=df_fair).fit(cov_type="HC1"),
            smf.ols(see_formula_3, data=df_fair).fit(cov_type="HC1"),
        ]
        total_models = [
            smf.ols(total_formula_1, data=df_fair).fit(cov_type="HC1"),
            smf.ols(total_formula_2, data=df_fair).fit(cov_type="HC1"),
            smf.ols(total_formula_3, data=df_fair).fit(cov_type="HC1"),
        ]

        all_results[metric] = see_models
        all_results_total[metric] = total_models

        with open(output_dir / f"table_{metric}_see_cie_models.txt", "w", encoding="utf-8") as handle:
            handle.write(
                model_summary_text(
                    see_models,
                    ["OLS", "OLS + City size FE", "OLS + Expansion x City size"],
                )
            )
        with open(output_dir / f"table_{metric}_total_expansion_models.txt", "w", encoding="utf-8") as handle:
            handle.write(
                model_summary_text(
                    total_models,
                    ["OLS", "OLS + City size FE", "OLS + TotalExpansion x City size"],
                )
            )

    see_table = build_total_effect_table(all_results, CITY_ORDER_4, "city_SEE")
    cie_table = build_total_effect_table(all_results, CITY_ORDER_4, "city_CIE")
    total_table = build_total_effect_table(all_results_total, CITY_ORDER_4, "city_TotalExpansion")

    see_table.to_csv(output_dir / "see_total_effects.csv", encoding="utf-8-sig")
    cie_table.to_csv(output_dir / "cie_total_effects.csv", encoding="utf-8-sig")
    total_table.to_csv(output_dir / "total_expansion_effects.csv", encoding="utf-8-sig")


def star_from_p(p_value: float | None) -> str:
    if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def find_interaction_name(params_index, var_prefix: str, city: str, fe_col: str = "city_level_4") -> str | None:
    matches = [
        item
        for item in params_index
        if var_prefix in item and f"[T.{city}]" in item and f"C({fe_col}" in item and ":" in item
    ]
    return matches[0] if matches else None


def total_effect_and_p(model, var_prefix: str, city: str, ref: str = "Medium/Small City", fe_col: str = "city_level_4"):
    params = model.params
    if var_prefix not in params.index:
        return None, None

    base_effect = float(params[var_prefix])
    if city == ref:
        return base_effect, float(model.pvalues[var_prefix])

    interaction_name = find_interaction_name(params.index, var_prefix, city, fe_col=fe_col)
    if interaction_name is None:
        return base_effect, None

    total_effect = base_effect + float(params[interaction_name])
    names = list(params.index)
    restriction = np.zeros((1, len(names)))
    restriction[0, names.index(var_prefix)] = 1.0
    restriction[0, names.index(interaction_name)] = 1.0
    test = model.wald_test(restriction)
    return total_effect, float(test.pvalue)


def build_total_effect_table(results_dict, city_levels: list[str], var_prefix: str) -> pd.DataFrame:
    rows = []
    for city in city_levels:
        row = {"City size": city}
        for metric, models in results_dict.items():
            effect, p_value = total_effect_and_p(models[2], var_prefix, city)
            row[metric] = "" if effect is None else f"{fmt_num(effect)}{star_from_p(p_value)}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("City size")


def lincomb(beta: pd.Series, cov: pd.DataFrame, terms: dict[str, float]) -> tuple[float, float]:
    terms = {key: value for key, value in terms.items() if key in beta.index}
    vector = pd.Series(0.0, index=beta.index)
    for key, weight in terms.items():
        vector.loc[key] = weight
    estimate = float((vector * beta).sum())
    variance = float(vector.values @ cov.values @ vector.values)
    return estimate, np.sqrt(variance)


def split_coef_star(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = df.replace(r"[^0-9\.\-]", "", regex=True).replace("", np.nan).astype(float)
    star = df.replace(r"[0-9\.\-]", "", regex=True)
    return numeric, star


def style_ax(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor("#f2f2f2")
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.6)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)


def remove_x_axis(ax) -> None:
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)


def plot_see_cie_effects(input_dir: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy import stats

    beta = pd.read_csv(input_dir / "model_acc3_params.csv", index_col=0)["coef"]
    cov = pd.read_csv(input_dir / "model_acc3_cov.csv", index_col=0)
    cov = cov.loc[beta.index, beta.index]
    df_resid = float(pd.read_csv(input_dir / "model_acc3_df.csv", index_col=0).iloc[0, 0])

    rows = []
    for city in CITY_ORDER_4:
        if city == "Medium/Small City":
            see_terms = {"city_SEE": 1.0}
            cie_terms = {"city_CIE": 1.0}
        else:
            see_int = f'city_SEE:C(city_level_4, Treatment(reference="Medium/Small City"))[T.{city}]'
            cie_int = f'city_CIE:C(city_level_4, Treatment(reference="Medium/Small City"))[T.{city}]'
            see_terms = {"city_SEE": 1.0, see_int: 1.0}
            cie_terms = {"city_CIE": 1.0, cie_int: 1.0}

        see_est, see_se = lincomb(beta, cov, see_terms)
        cie_est, cie_se = lincomb(beta, cov, cie_terms)
        rows.append([city, "SEE", see_est, see_se])
        rows.append([city, "CIE", cie_est, cie_se])

    marginal_effects = pd.DataFrame(rows, columns=["city_level", "path", "effect", "se"])
    tcrit = stats.t.ppf(0.975, df=df_resid)
    marginal_effects["ci_low"] = marginal_effects["effect"] - tcrit * marginal_effects["se"]
    marginal_effects["ci_high"] = marginal_effects["effect"] + tcrit * marginal_effects["se"]
    marginal_effects["p"] = 2 * (
        1 - stats.t.cdf(np.abs(marginal_effects["effect"] / marginal_effects["se"]), df=df_resid)
    )
    marginal_effects["star"] = marginal_effects["p"].apply(star_from_p)

    see_raw = pd.read_csv(input_dir / "see_total_effects.csv", index_col=0).loc[CITY_ORDER_4, METRIC_ORDER]
    cie_raw = pd.read_csv(input_dir / "cie_total_effects.csv", index_col=0).loc[CITY_ORDER_4, METRIC_ORDER]
    total_raw = pd.read_csv(input_dir / "total_expansion_effects.csv", index_col=0).loc[CITY_ORDER_4, METRIC_ORDER]

    see_num, see_star = split_coef_star(see_raw)
    cie_num, cie_star = split_coef_star(cie_raw)
    total_num, total_star = split_coef_star(total_raw)
    see_sig = see_star.applymap(lambda value: isinstance(value, str) and ("*" in value))
    cie_sig = cie_star.applymap(lambda value: isinstance(value, str) and ("*" in value))
    total_sig = total_star.applymap(lambda value: isinstance(value, str) and ("*" in value))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(9.6, 12))
    grid = GridSpec(nrows=3, ncols=4, figure=fig, height_ratios=[0.8, 1.0, 1.0], hspace=0.10, wspace=0.08)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.10)
    axes_map = {}

    x_pos = np.array([0, 1])
    paths = ["SEE", "CIE"]
    colors = ["#DD8452", "#385a7a"]
    markers = {"SEE": "o", "CIE": "^"}

    for column_idx, (city, title) in enumerate(zip(CITY_ORDER_4, CITY_LABELS)):
        ax = fig.add_subplot(grid[0, column_idx])
        axes_map[(0, column_idx)] = ax
        style_ax(ax)
        ax.set_title(title)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-1.9, 9.5)
        temp = marginal_effects[marginal_effects["city_level"] == city].set_index("path").loc[paths]

        for idx, path_name in enumerate(paths):
            effect = float(temp.loc[path_name, "effect"])
            low = float(temp.loc[path_name, "ci_low"])
            high = float(temp.loc[path_name, "ci_high"])
            yerr = np.array([[effect - low], [high - effect]])
            is_sig = (low > 0) or (high < 0)
            ax.errorbar(
                x_pos[idx],
                effect,
                yerr=yerr,
                fmt=markers[path_name],
                color=colors[idx],
                ecolor=colors[idx],
                capsize=6,
                elinewidth=1.8,
                markersize=8,
                markeredgewidth=1.6,
                linewidth=1.8,
                markerfacecolor=(colors[idx] if is_sig else "none"),
                markeredgecolor=colors[idx],
            )
            if is_sig:
                dx = 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                ax.text(
                    x_pos[idx] + dx,
                    effect,
                    f"{effect:.2f}{temp.loc[path_name, 'star']}",
                    ha="left",
                    va="center",
                    fontsize=12,
                )

        if column_idx == 0:
            ax.legend(loc="upper left", frameon=False, handlelength=2.2)
            ax.set_ylabel("Marginal effect on accessibility")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        remove_x_axis(ax)

    def plot_single_bar_row(row_idx: int, values_df, stars_df, sig_df, y_label: str, ylim: tuple[float, float]) -> None:
        for column_idx, city in enumerate(CITY_ORDER_4):
            ax = fig.add_subplot(grid[row_idx, column_idx])
            axes_map[(row_idx, column_idx)] = ax
            style_ax(ax)
            ax.set_ylim(ylim)
            x = np.arange(len(METRIC_ORDER))
            vals = values_df.loc[city, METRIC_ORDER].values
            stars = stars_df.loc[city, METRIC_ORDER].values
            sigs = sig_df.loc[city, METRIC_ORDER].values

            for idx, metric in enumerate(METRIC_ORDER):
                value = vals[idx]
                if np.isnan(value):
                    continue
                alpha = 1.0 if sigs[idx] else 0.25
                ax.bar(x[idx], value, width=0.42, color=METRIC_COLORS[metric], alpha=alpha, edgecolor="black", linewidth=1.0)
                dy = 0.03 * (ylim[1] - ylim[0])
                if sigs[idx]:
                    ax.text(
                        x[idx],
                        value + (dy if value >= 0 else -dy),
                        f"{value:.2f}{stars[idx]}",
                        ha="center",
                        va="bottom" if value >= 0 else "top",
                        fontsize=12,
                    )

            if column_idx == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            remove_x_axis(ax)

    def plot_combined_row(row_idx: int, see_values, see_stars, see_sig_mask, cie_values, cie_stars, cie_sig_mask, y_label: str, ylim: tuple[float, float]) -> None:
        x = np.arange(len(METRIC_ORDER))
        bar_width = 0.16
        offset = 0.14
        for column_idx, city in enumerate(CITY_ORDER_4):
            ax = fig.add_subplot(grid[row_idx, column_idx])
            axes_map[(row_idx, column_idx)] = ax
            style_ax(ax)
            ax.set_ylim(ylim)

            for idx, metric in enumerate(METRIC_ORDER):
                face = METRIC_COLORS[metric]
                for xpos, value, sig, stars, marker in [
                    (x[idx] - offset, see_values.loc[city, metric], see_sig_mask.loc[city, metric], see_stars.loc[city, metric], "s"),
                    (x[idx] + offset, cie_values.loc[city, metric], cie_sig_mask.loc[city, metric], cie_stars.loc[city, metric], "^"),
                ]:
                    if np.isnan(value):
                        continue
                    alpha = 1.0 if sig else 0.25
                    ax.bar(xpos, value, width=bar_width, color=face, alpha=alpha, edgecolor="black", linewidth=1.0, zorder=2)
                    ax.plot(
                        xpos,
                        value,
                        marker=marker,
                        markersize=6.0,
                        markerfacecolor="black",
                        markeredgecolor="black",
                        markeredgewidth=1.0,
                        linestyle="None",
                        alpha=alpha,
                        zorder=3,
                    )
                    if sig:
                        dy = 0.03 * (ylim[1] - ylim[0])
                        ax.text(
                            xpos,
                            value + (dy if value >= 0 else -dy),
                            f"{value:.2f}{stars}",
                            ha="center",
                            va="bottom" if value >= 0 else "top",
                            fontsize=12,
                        )

            if column_idx == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            remove_x_axis(ax)

    plot_single_bar_row(1, total_num, total_star, total_sig, "Total effect on inequality", (-0.45, 0.30))
    plot_combined_row(2, see_num, see_star, see_sig, cie_num, cie_star, cie_sig, "Pathway effects on inequality", (-0.45, 0.30))

    metric_legend = [
        Patch(facecolor=METRIC_COLORS["gini"], edgecolor="black", label="Gini"),
        Patch(facecolor=METRIC_COLORS["theil"], edgecolor="black", label="Theil"),
        Patch(facecolor=METRIC_COLORS["mld"], edgecolor="black", label="MLD"),
    ]
    axes_map[(1, 0)].legend(handles=metric_legend, loc="upper left", frameon=False, handlelength=2.2, columnspacing=1.6)

    path_legend = [
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="black", markeredgecolor="black", markeredgewidth=1.0, markersize=6.5, label="SEE"),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="black", markeredgecolor="black", markeredgewidth=1.0, markersize=6.5, label="CIE"),
    ]
    axes_map[(2, 0)].legend(handles=path_legend, loc="upper left", frameon=False, handlelength=1.2, columnspacing=1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()
