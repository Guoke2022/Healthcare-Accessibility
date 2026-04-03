from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_stats(csv_path: Path, year_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Load the year column and target value columns from one statistics table."""
    df = pd.read_csv(csv_path)
    return df[[year_col] + value_cols]


def build_shapley_scenarios(baseline_df, shapley_df, year_col, baseline_year, target_year, scenario_map):
    """Assemble the 2^3 scenario table required for three-factor Shapley decomposition."""
    scenarios = {"A000": baseline_df.loc[baseline_df[year_col] == baseline_year].iloc[0], "A111": baseline_df.loc[baseline_df[year_col] == target_year].iloc[0]}
    for _, row in shapley_df.iterrows():
        key = scenario_map.get(row[year_col])
        if key is not None:
            scenarios[key] = row
    if len(scenarios) != 8:
        raise ValueError("Incomplete Shapley scenarios. Please check the input files and labels.")
    return scenarios


def shapley_3factors(scenarios, var):
    """Compute factor contributions for one outcome under a three-factor design."""
    a = scenarios
    phi_r = ((a["A100"][var] - a["A000"][var]) + (a["A110"][var] - a["A010"][var]) + (a["A101"][var] - a["A001"][var]) + (a["A111"][var] - a["A011"][var])) / 4
    phi_p = ((a["A010"][var] - a["A000"][var]) + (a["A110"][var] - a["A100"][var]) + (a["A011"][var] - a["A001"][var]) + (a["A111"][var] - a["A101"][var])) / 4
    phi_b = ((a["A001"][var] - a["A000"][var]) + (a["A101"][var] - a["A100"][var]) + (a["A011"][var] - a["A010"][var]) + (a["A111"][var] - a["A110"][var])) / 4
    total = a["A111"][var] - a["A000"][var]
    return {"Road": phi_r, "Population": phi_p, "Bed": phi_b, "Total_change": total}


def shapley_postprocess(shap_dict):
    """Scale raw Shapley contributions to match the observed total change exactly."""
    contrib = pd.Series({"Road": shap_dict["Road"], "Population": shap_dict["Population"], "Bed": shap_dict["Bed"]})
    total = shap_dict["Total_change"]
    contrib_adj = contrib * (total / contrib.sum())
    pct = contrib_adj / total * 100
    return pd.DataFrame({"Contribution_abs": contrib_adj, "Contribution_pct": pct})


def run_shapley_tasks(tasks, scenario_map, output_dir: Path, year_col: str = "Year", baseline_year: int = 2014, target_year: int = 2024):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for task in tasks:
        df_base = load_stats(Path(task["baseline_csv"]), year_col, task["value_cols"])
        df_shap = load_stats(Path(task["shapley_csv"]), year_col, task["value_cols"])
        scenarios = build_shapley_scenarios(df_base, df_shap, year_col, baseline_year, target_year, scenario_map)
        for var in task["value_cols"]:
            shap_raw = shapley_3factors(scenarios, var)
            df_res = shapley_postprocess(shap_raw)
            df_res["task"] = task["name"]
            df_res["indicator"] = var
            a000_row = pd.DataFrame({"Contribution_abs": [scenarios["A000"][var]], "Contribution_pct": [0.0], "task": [task["name"]], "indicator": [var], "factor": [str(baseline_year)]})
            a111_row = pd.DataFrame({"Contribution_abs": [scenarios["A111"][var]], "Contribution_pct": [100.0], "task": [task["name"]], "indicator": [var], "factor": [str(target_year)]})
            df_res_shap = df_res.reset_index(names="factor")
            df_res_final = pd.concat([a000_row, df_res_shap, a111_row], ignore_index=True)
            all_results.append(df_res_final)
            df_res_final.to_csv(output_dir / f"shapley_{task['name']}_{var}.csv", index=False)
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(output_dir / "shapley_all_results.csv", index=False)
    return df_all


def plot_shapley_waterfalls(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [("accessibility", "pop_median", "acc_pop_median"), ("inequality_acc", "pop_gini", "ineq_pop_gini"), ("inequality_acc", "pop_theil", "ineq_pop_theil"), ("inequality_acc", "pop_MLD", "ineq_pop_MLD")]
    title_map = {"pop_median": "Median accessibility", "pop_gini": "Gini", "pop_theil": "Theil", "pop_MLD": "MLD"}
    improve_sign = {"accessibility": +1, "inequality_acc": -1}
    factor_order = ["2014", "Bed", "Population", "Road", "2024"]
    factor_label = {"Bed": "Hospital\nexpansion", "Population": "Population\nchange", "Road": "Road\nupgrades"}

    for task, indicator, stem in plots:
        subset = df[(df["task"] == task) & (df["indicator"] == indicator)].copy()
        if subset.empty:
            continue
        subset = subset.set_index("factor").reindex(factor_order)
        values = subset["Contribution_abs"].to_dict()
        pct = subset["Contribution_pct"].to_dict()
        start = float(values["2014"])
        end = float(values["2024"])
        deltas = {"Bed": float(values["Bed"]), "Population": float(values["Population"]), "Road": float(values["Road"])}

        positions = {"2014": start}
        current = start
        for factor in ["Bed", "Population", "Road"]:
            positions[factor] = (current, current + deltas[factor])
            current += deltas[factor]
        positions["2024"] = end
        colors = {factor: ("#E6862E" if improve_sign[task] * deltas[factor] >= 0 else "#4E79A7") for factor in deltas}

        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.axvline(start, color="#8a8a8a", linewidth=1.2)
        ax.axvline(end, color="#8a8a8a", linewidth=1.2)
        ax.scatter([start, end], [2.58, 2.58], s=30, color="#8a8a8a", zorder=4)
        ax.text(start, 2.70, "2014", ha="center", va="bottom", fontsize=12)
        ax.text(end, 2.45, "2024", ha="center", va="top", fontsize=12)

        for y, factor in zip([2.0, 1.0, 0.0], ["Bed", "Population", "Road"]):
            x0, x1 = positions[factor]
            ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=colors[factor], mutation_scale=8))
            ax.text((x0 + x1) / 2, y + 0.08, f"{pct[factor]:.1f}%", ha="center", va="bottom", fontsize=14, color=colors[factor])
            ax.text((x0 + x1) / 2, y - 0.18, factor_label[factor], ha="center", va="top", fontsize=12)

        ax.set_title(title_map[indicator], fontsize=15)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
        plt.close()
