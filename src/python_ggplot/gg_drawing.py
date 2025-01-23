from typing import Any, Dict, Generator, List, Tuple, cast

import numpy as np
import pandas as pd

from python_ggplot.core.coord.objects import Coord1D, RelativeCoordType
from python_ggplot.core.objects import AxisKind, GGException
from python_ggplot.core.units.objects import Quantity, RelativeUnit, UnitType
from python_ggplot.datamancer_pandas_compat import GGValue, VNull
from python_ggplot.gg_styles import GGStyle
from python_ggplot.gg_types import (
    BinPositionType,
    DiscreteType,
    FilledGeom,
    FilledGeomDiscrete,
    OutsideRangeKind,
    Theme,
)
from python_ggplot.graphics.draw import layout
from python_ggplot.graphics.initialize import init_coord_1d
from python_ggplot.graphics.views import ViewPort


def is_num(x):
    if isinstance(x, (int, float)):
        return True
    if not getattr(x, "dtype", None):
        return False
    return np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer)


def enumerate_data(
    geom: FilledGeom,
) -> Generator[Tuple[GGValue, GGStyle, List[GGStyle], pd.DataFrame]]:
    for label, tup in geom.yield_data.items():
        yield label, tup[0], tup[1], tup[2]


def get_xy(x_t, y_t, i):
    x = 0.0 if pd.isna(x_t[i]) else x_t[i]
    y = 0.0 if pd.isna(y_t[i]) else y_t[i]

    # there was a check using %~ which i think it ensure that
    # we dont have float32 and float64, so the rest of the logic may not be needed

    return (x, y)


def read_or_calc_bin_width(df, idx, data_col, dc_kind: DiscreteType, col="binWidths"):
    # TODO clean up later
    if dc_kind == DiscreteType.CONTINUOUS:
        if col in df.columns:
            if pd.isna(df.iloc[idx][col]):
                return None
            return df.iloc[idx][col]
        elif idx < len(df) - 1:
            high_val = df.iloc[idx + 1][data_col]
            if pd.isna(high_val):
                if idx <= 0:
                    raise GGException("expected idx> 0")
                return df.iloc[idx][data_col] - df.iloc[idx - 1][data_col]
            else:
                return high_val - df.iloc[idx][data_col]
    elif dc_kind == DiscreteType.DISCRETE:
        return 0.8

    raise GGException()


def move_bin_position(x, bp_kind: BinPositionType, bin_width):
    lookup = {
        BinPositionType.LEFT: x,
        BinPositionType.NONE: x,
        BinPositionType.CENTER: x + (bin_width / 2.0),
        BinPositionType.RIGHT: x + bin_width,
    }
    return lookup[bp_kind]


def read_error_data(df, idx, fg):
    result = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}

    if fg.x_min is not None:
        result["x_min"] = df[fg.x_min].iloc[idx]
    if fg.x_max is not None:
        result["x_max"] = df[fg.x_max].iloc[idx]
    if fg.y_min is not None:
        result["y_min"] = df[fg.y_min].iloc[idx]
    if fg.y_max is not None:
        result["y_max"] = df[fg.y_max].iloc[idx]

    return result["x_min"], result["x_max"], result["y_min"], result["y_max"]


def read_width_height(df, idx, fg):
    width = 1.0
    height = 1.0

    if fg.width is not None:
        width = df[fg.width].iloc[idx]

    if fg.height is not None:
        height = df[fg.height].iloc[idx]

    return width, height


def read_text(df, idx, fg):
    return df[fg.text].iloc[idx]


def get_cols_and_rows(fg: FilledGeom) -> Tuple[int, int]:
    cols, rows = 1, 1

    if fg.x_discrete_kind.discrete_type == DiscreteType.DISCRETE:
        f_geom_ = cast(FilledGeomDiscrete, fg.x_discrete_kind)
        cols = len(f_geom_.label_seq)

    if fg.y_discrete_kind.discrete_type == DiscreteType.DISCRETE:
        f_geom_ = cast(FilledGeomDiscrete, fg.x_discrete_kind)
        rows = len(f_geom_.label_seq)

    return cols, rows


def prepare_views(view: ViewPort, fg: FilledGeom, theme: Theme):

    cols, rows = get_cols_and_rows(fg)

    discrete_margin = theme.discrete_scale_margin or 0.0

    widths = []
    heights = []

    if cols > 1:
        ind_widths = [0.0] * cols
        cols += 2
        widths = [discrete_margin] + ind_widths + [discrete_margin]

    if rows > 1:
        ind_heights = [0.0] * rows
        rows += 2
        heights = [discrete_margin] + ind_heights + [discrete_margin]

    widths_q = [RelativeUnit(i) for i in widths]
    heights_q = [RelativeUnit(i) for i in heights]

    # TODO low priority maybe we change quantity to protocol to remove cast?

    layout(
        view,
        cols=cols,
        rows=rows,
        col_widths=cast(List[Quantity], widths_q),
        row_heights=cast(List[Quantity], heights_q),
    )


def calc_view_map(fg: FilledGeom) -> Dict[Any, Any]:
    result: Dict[Any, Any] = {}
    cols, rows = get_cols_and_rows(fg)

    if cols == 1 and rows == 1:
        # not discrete
        return result
    elif rows == 1 and cols > 1:
        y = VNull()
        for j in range(cols):
            x = fg.get_x_label_seq()[j]
            result[(x, y)] = j + 1
    elif cols == 1 and rows > 1:
        x = VNull()
        for i in range(rows):
            y = fg.get_y_label_seq()[i]
            result[(x, y)] = i + 1
    else:
        for i in range(rows):
            y = fg.get_y_label_seq()[i]
            for j in range(cols):
                x = fg.get_x_label_seq()[j]
                result[(x, y)] = (i + 1) * (cols + 2) + (j + 1)

    return result


def get_discrete_histogram(width: float, ax_kind: AxisKind) -> Coord1D:
    if ax_kind == AxisKind.X:
        left = (1.0 - width) / 2.0

        return init_coord_1d(left, AxisKind.X, UnitType.RELATIVE)
    else:
        top = 1.0
        return RelativeCoordType(top)
