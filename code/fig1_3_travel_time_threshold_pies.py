# -*- coding = utf-8 -*-
# @Author ：YEPEI
# @Time : 2025/6/2 17:18

# @Software : PyCharm

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from config import PLOT_DPI
from utils.figure_data import figure_dir, load_admin_level, time_threshold_frame

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.2f}'.format)


LABEL_FONTSIZE = 25
TICKS_FONTSIZE = 20
LEGEND_FONTSIZE = 25

THRESHOLD_MIN = 60


CLASS_LABELS = {
    1: '2014_le60',
    2: 'new_2024',
    3: '2024_gt60',
}


def clean_admin_frame(df: pd.DataFrame, join_key: str, scale_name: str) -> pd.DataFrame:
    """Remove rows that do not represent a valid administrative unit.

    ``pandas.merge`` can match null keys across years, so a row with a missing
    province/city/county key may survive ``time_threshold_frame`` and otherwise
    be counted as if it were a real administrative unit.  Fig. 1c-e must count
    only named/coded administrative units that can be mapped.
    """
    out = df.copy()

    key = out[join_key].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)

    # Province/city keys only need to be non-empty. County statistics must also
    # exclude placeholder/malformed codes such as 000000; these are aggregate or
    # non-mappable records rather than county analysis units. Filtering happens
    # here, before the Fig.1e counts are calculated, so the inset pie and the
    # exported polygon layer always use the same county universe.
    invalid = key.isna() | key.eq('')
    if join_key == '县级码':
        key = key.str.zfill(6)
        invalid = invalid | ~key.str.fullmatch(r'\d{6}', na=False) | key.eq('000000')

    out = out.loc[~invalid].copy()
    out[join_key] = key.loc[~invalid]

    if out[join_key].duplicated().any():
        dup = out.loc[out[join_key].duplicated(keep=False), [join_key, '14_time', '24_time']]
        raise RuntimeError(
            f"{scale_name} 统计表存在重复行政区 key，无法安全生成 Fig.1 图层：\n"
            f"{dup.to_string(index=False)}"
        )

    return out


def add_threshold_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three Fig. 1c-e threshold classes to a 2014/2024 frame."""
    out = df.copy()

    out['class_id'] = 3
    out.loc[out['14_time'] <= THRESHOLD_MIN, 'class_id'] = 1
    out.loc[
        (out['14_time'] > THRESHOLD_MIN) & (out['24_time'] <= THRESHOLD_MIN),
        'class_id'
    ] = 2

    out['class60'] = out['class_id'].map(CLASS_LABELS)
    return out


def plot_pie(df: pd.DataFrame, output_path: str, scale_name: str) -> None:
    """Plot the inset pie for the three travel-time threshold classes."""
    classified = add_threshold_class(df)

    num_2014 = int((classified['class_id'] == 1).sum())
    new_in_2024 = int((classified['class_id'] == 2).sum())
    remaining_2024 = int((classified['class_id'] == 3).sum())


    sizes = [remaining_2024, new_in_2024, num_2014]
    colors = ['#f2f3f2', '#cd1e71', '#b2d584']

    plt.figure(figsize=(4, 4))
    plt.pie(
        sizes,
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white'}
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=PLOT_DPI, transparent=True, bbox_inches='tight')
    plt.close()


def load_fig1_admin_level(admin_level: str):
    """Helper for load_fig1_admin_level."""


    if admin_level != 'city':
        return load_admin_level(admin_level)

    county = load_admin_level('county').copy()
    required = ['县级', '地级', '省级']
    missing_cols = [c for c in required if c not in county.columns]
    if missing_cols:
        raise KeyError(
            f"Fig.1d city geometry requires county-boundary columns {required}; "
            f"missing={missing_cols}"
        )

    for col in required:
        county[col] = county[col].astype('string').str.strip()

    city_key = county['地级'].copy()

    # Keep this logic synchronized with add_city_level() in
    # 2_1_multiscale_match.py.
    fallback_to_province = city_key.isna() | city_key.eq('') | city_key.eq('不统计')
    city_key = city_key.mask(fallback_to_province, county['省级'])

    directly_administered = city_key.isin(['海南省', '湖北省'])
    city_key = city_key.mask(directly_administered, county['县级'])

    still_missing = city_key.isna() | city_key.eq('')
    city_key = city_key.mask(still_missing, county['县级'])

    tmp = county.copy()
    tmp['地级'] = city_key

    invalid = tmp['地级'].isna() | tmp['地级'].eq('')
    tmp = tmp.loc[~invalid].copy()

    out = tmp.dissolve(by='地级', as_index=False)
    return out

def export_arcgis_layer(
    df: pd.DataFrame,
    admin_level: str,
    join_key: str,
    output_path: str,
    scale_name: str,
) -> None:
    """Export a Fig. 1c-e administrative polygon layer ready for ArcGIS.

    Output fields added to the administrative boundary layer:
      t14_med  : population-weighted median travel time in 2014 (min)
      t24_med  : population-weighted median travel time in 2024 (min)
      class_id : 1 / 2 / 3 threshold class
      class60  : 2014_le60 / new_2024 / 2024_gt60

    Only units present in the travel-time statistics are retained, matching the
    population used by the inset pie counts.
    """
    classified = add_threshold_class(df)
    attrs = classified[[join_key, '14_time', '24_time', 'class_id', 'class60']].rename(
        columns={
            '14_time': 't14_med',
            '24_time': 't24_med',
        }
    )

    admin = load_fig1_admin_level(admin_level)

    # Normalize key types before merging. County codes need six digits.
    admin[join_key] = admin[join_key].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    attrs[join_key] = attrs[join_key].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)

    if join_key == '县级码':
        admin[join_key] = admin[join_key].str.zfill(6)
        attrs[join_key] = attrs[join_key].str.zfill(6)

        # The fixed county-boundary layer contains repeated placeholder records

        # not represent county analysis units and are excluded upstream from the
        # county statistics, so they must not participate in the Fig.1e join.
        admin_valid = admin[join_key].str.fullmatch(r'\d{6}', na=False) & admin[join_key].ne('000000')
        admin = admin.loc[admin_valid].copy()

        # A county may be represented by multiple polygon records (e.g. detached
        # parts/islands).  Fig.1e needs one analysis unit per county code, so
        # dissolve duplicate valid codes instead of treating them as duplicate
        # statistical units.
        dup_mask = admin[join_key].duplicated(keep=False)
        if dup_mask.any():
            admin = admin.dissolve(by=join_key, as_index=False)

        # Defensive check: clean_admin_frame() should already have removed
        # invalid county codes before classification/counting.
        attr_valid = attrs[join_key].str.fullmatch(r'\d{6}', na=False) & attrs[join_key].ne('000000')
        if (~attr_valid).any():
            bad = attrs.loc[~attr_valid, [join_key, 't14_med', 't24_med']].copy()
            raise RuntimeError(
                f"{scale_name} 统计表清洗后仍含无效县级码（程序逻辑异常）：\n"
                f"{bad.to_string(index=False)}"
            )

    # At this point every map key used for a one-to-one join must be unique.
    if admin[join_key].duplicated().any():
        dup = admin.loc[admin[join_key].duplicated(keep=False), [join_key]].copy()
        raise RuntimeError(
            f"{scale_name} 空间图层在清理后仍存在重复 join key：\n"
            f"{dup.head(30).to_string(index=False)}"
        )

    stat_keys = set(attrs[join_key].dropna().tolist())
    admin_keys = set(admin[join_key].dropna().tolist())
    missing_in_map = sorted(stat_keys - admin_keys)
    if missing_in_map:
        raise RuntimeError(
            f"{scale_name} 有 {len(missing_in_map)} 个统计单位在行政区图层中找不到："
            f"{missing_in_map[:20]}"
        )

    layer = admin.merge(attrs, on=join_key, how='inner', validate='one_to_one')

    if len(layer) != len(attrs):
        raise RuntimeError(
            f"{scale_name} ArcGIS 图层匹配数量异常：有效统计单位 {len(attrs)} 个，"
            f"空间图层匹配 {len(layer)} 个。"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    layer.to_file(output_path, encoding='utf-8')


def main() -> None:
    fig1_dir = figure_dir('Figure 1')
    arcgis_dir = figure_dir('Map_layers', 'Fig1_travel_time_60min')
    os.makedirs(arcgis_dir, exist_ok=True)

    jobs = [
        {
            'group': 'provincial',
            'stats_key': '省级',
            'admin_level': 'province',
            'join_key': '省级',
            'scale_name': 'Province',
            'pie_name': 'province_60min_pie.png',
            'shp_name': 'Fig1c_province_60min.shp',
        },
        {
            'group': 'city',
            'stats_key': '地级',
            'admin_level': 'city',
            'join_key': '地级',
            'scale_name': 'City',
            'pie_name': 'city_60min_pie.png',
            'shp_name': 'Fig1d_city_60min.shp',
        },
        {
            'group': 'county',
            'stats_key': '县级码',
            'admin_level': 'county',
            'join_key': '县级码',
            'scale_name': 'County',
            'pie_name': 'county_60min_pie.png',
            'shp_name': 'Fig1e_county_60min.shp',
        },
    ]

    for job in jobs:
        frame = time_threshold_frame(job['group'], job['stats_key'])
        frame = clean_admin_frame(frame, job['join_key'], job['scale_name'])

        plot_pie(
            frame,
            str(fig1_dir / job['pie_name']),
            job['scale_name'],
        )

        export_arcgis_layer(
            frame,
            job['admin_level'],
            job['join_key'],
            str(arcgis_dir / job['shp_name']),
            job['scale_name'],
        )


if __name__ == '__main__':
    main()
