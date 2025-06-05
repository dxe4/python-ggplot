from copy import deepcopy
from itertools import product
from typing import Any, Generator, List, Tuple

import pandas as pd

from python_ggplot.core.objects import Style
from python_ggplot.gg.geom.base import FilledGeom, FilledStatGeom
from python_ggplot.gg.types import GGStyle


def enumerate_groups(
    df: pd.DataFrame, columns: List[str]
) -> Generator[Tuple[Tuple[Any, ...], pd.DataFrame, Any], None, None]:
    grouped = df.groupby(columns, sort=True)  # type: ignore
    sorted_keys = sorted(grouped.groups.keys(), reverse=True)  # type: ignore
    for keys in sorted_keys:  # type: ignore
        sub_df = grouped.get_group(keys)  # type: ignore
        key_values = list(product(columns, [keys]))  # type: ignore
        yield (keys, sub_df, key_values)


def maybe_inherit_aes(
    sub_df: pd.DataFrame,
    filled_stat_geom: FilledStatGeom,
    filled_geom: FilledGeom,
    style: GGStyle,
    key_values: Any,
) -> GGStyle:
    from python_ggplot.gg.styles.utils import apply_style

    style = deepcopy(style)

    if filled_geom.gg_data.geom.inherit_aes():
        return apply_style(
            style, sub_df, filled_stat_geom.discrete_scales, key_values
        )  # type: ignore
    else:
        return style
