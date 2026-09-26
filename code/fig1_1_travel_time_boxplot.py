# -*- coding = utf-8 -*-

import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings
from config import PLOT_DPI
from utils.figure_data import read_stats, figure_dir
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.2f}'.format)

def main() -> None:

    df = read_stats('national', 'travel_time')

    LABEL_FONTSIZE = 23 # 25
    TICKS_FONTSIZE = 18 # 20

    boxplot_data = []
    for _, row in df.iterrows():
        q1 = row['pop_25%']
        med = row['pop_median']
        q3 = row['pop_75%']
        iqr = q3 - q1

        lower_whisker = 1
        upper_whisker = q3 + 1.5 * iqr

        box = {
            'med': med,
            'q1': q1,
            'q3': q3,
            'whislo': lower_whisker,
            'whishi': upper_whisker,
            'fliers': []
        }
        boxplot_data.append(box)


    fig, ax = plt.subplots(figsize=(20, 6))

    bp = ax.bxp(
        boxplot_data,
        positions=range(len(df)),
        showfliers=False,
        patch_artist=True,
        boxprops=dict(color='white'),
        medianprops=dict(color="white", linewidth=4),
        whiskerprops=dict(color="gray", linewidth=3),
        capprops=dict(color="gray", linewidth=3)
    )


    for patch in bp['boxes']:
        patch.set_facecolor("#b1cc8e")
        patch.set_alpha(0.5)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Year'], fontsize=TICKS_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICKS_FONTSIZE)
    ax.set_xlabel("Year", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Travel time(min)", fontsize=LABEL_FONTSIZE)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.tight_layout()

    output_dir = str(figure_dir('Figure 1'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'figure1a.png')
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
