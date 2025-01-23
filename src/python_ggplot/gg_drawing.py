from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from python_ggplot.core.coord.objects import (
    Coord,
    Coord1D,
    DataCoordType,
    RelativeCoordType,
)
from python_ggplot.core.objects import AxisKind, GGException, Style
from python_ggplot.core.units.objects import DataUnit, Quantity, RelativeUnit, UnitType
from python_ggplot.datamancer_pandas_compat import GGValue, VNull
from python_ggplot.gg_geom import (
    FilledGeom,
    FilledGeomDiscrete,
    FilledGeomErrorBar,
    FilledGeomRaster,
    GeomType,
    HistogramDrawingStyle,
)
from python_ggplot.gg_styles import GGStyle
from python_ggplot.gg_types import BinPositionType  # OutsideRangeKind,
from python_ggplot.gg_types import DiscreteType, PositionType, Theme
from python_ggplot.graphics.draw import layout
from python_ggplot.graphics.initialize import (
    InitErrorBarData,
    InitRasterData,
    InitRectInput,
    InitTextInput,
    init_coord_1d,
    init_error_bar,
    init_point,
    init_raster,
    init_rect,
    init_text,
)
from python_ggplot.graphics.objects import GOComposite
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


def read_or_calc_bin_width(
    df: pd.DataFrame, idx: int, data_col: str, dc_kind: DiscreteType, col="binWidths"
):
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


def read_error_data(
    df: pd.DataFrame, idx: int, fg: FilledGeomErrorBar
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
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
            x_ = fg.x_discrete_kind.get_label_seq()[i]
            result[(x_, y)] = i + 1
    else:
        for i in range(rows):
            y_ = fg.y_discrete_kind.get_label_seq()[i]
            for j in range(cols):
                x = fg.get_x_label_seq()[j]
                result[(x, y_)] = (i + 1) * (cols + 2) + (j + 1)

    return result


def get_discrete_histogram(width: float, ax_kind: AxisKind) -> Coord1D:
    if ax_kind == AxisKind.X:
        left = (1.0 - width) / 2.0

        return init_coord_1d(left, AxisKind.X, UnitType.RELATIVE)
    else:
        # TODO high priority double check this.
        # templace c1 in ginger has default AxisKind.X which is used in this case
        top = 1.0
        return init_coord_1d(top, AxisKind.X, UnitType.RELATIVE)


def get_discrete_point() -> Coord1D:
    return RelativeCoordType(0.5)


def get_discrete_line(view: ViewPort, axis_kind: AxisKind) -> Coord1D:
    center_x, center_y = view.get_center()
    lookup = {
        AxisKind.X: init_coord_1d(center_x, AxisKind.X, UnitType.RELATIVE),
        AxisKind.Y: init_coord_1d(center_y, AxisKind.X, UnitType.RELATIVE),
    }

    return lookup[axis_kind]


def get_continuous(
    view: ViewPort, fg: FilledGeom, val: float, ax_kind: AxisKind
) -> Coord1D:
    scale_lookup = {AxisKind.X: view.x_scale, AxisKind.Y: view.y_scale}
    scale = scale_lookup[ax_kind]
    if scale is None:
        raise GGException("expected a scale")
    return Coord1D.create_data(val, scale, ax_kind)


def get_draw_pos_impl(
    view: ViewPort,
    fg: FilledGeom,
    val,
    width: float,
    discrete_type: DiscreteType,
    ax_kind: AxisKind,
):

    fg_geom_type = fg.geom_type()
    if discrete_type == DiscreteType.DISCRETE:
        if fg_geom_type in {GeomType.POINT, GeomType.ERROR_BAR, GeomType.TEXT}:
            return get_discrete_point()
        elif fg_geom_type in (GeomType.LINE, GeomType.FREQ_POLY):
            return get_discrete_line(view, ax_kind)
        elif fg_geom_type in (GeomType.HISTOGRAM, GeomType.BAR):
            return get_discrete_histogram(width, ax_kind)
        elif fg_geom_type == GeomType.TILE:
            return get_discrete_histogram(width, ax_kind)
        elif fg_geom_type == GeomType.RASTER:
            return get_discrete_histogram(1.0, ax_kind)

    elif discrete_type == DiscreteType.CONTINUOUS:
        if fg_geom_type in {
            GeomType.POINT,
            GeomType.ERROR_BAR,
            GeomType.LINE,
            GeomType.FREQ_POLY,
            GeomType.HISTOGRAM,
            GeomType.BAR,
            GeomType.TILE,
            GeomType.RASTER,
            GeomType.TEXT,
        }:
            return get_continuous(view, fg, val, ax_kind)
    else:
        raise GGException("unknown discrete type")


def get_draw_pos(
    view: ViewPort,
    view_idx: int,
    fg: FilledGeom,
    p: Tuple[float, float],
    bin_widths: Tuple[float, float],
    df: pd.DataFrame,
    idx: int,
):

    coords_flipped = False

    geom_type = fg.geom_type()
    position = fg.geom.position
    histogram_drawing_style = fg.geom.histogram_drawing_style

    if position == PositionType.IDENTITY:
        mp = list(p)
        if geom_type == GeomType.BAR or (
            geom_type == GeomType.HISTOGRAM
            and histogram_drawing_style == HistogramDrawingStyle.BARS
        ):
            if not coords_flipped:
                mp[1] = 0.0
            else:
                mp[0] = 0.0

        result_x = get_draw_pos_impl(
            view,
            fg,
            mp[0],
            bin_widths[0],
            fg.x_discrete_kind.discrete_type(),
            AxisKind.X,
        )
        result_y = get_draw_pos_impl(
            view,
            fg,
            mp[1],
            bin_widths[1],
            fg.y_discrete_kind.discrete_type(),
            AxisKind.Y,
        )

    elif position == PositionType.STACK:
        if not (
            (
                fg.geom.kind == "histogram"
                and histogram_drawing_style == HistogramDrawingStyle.BARS
            )
            or fg.geom.kind == "bar"
        ):
            cur_stack = p[1]
        else:
            cur_stack = df["PrevVals"].iloc[idx]

        if not coords_flipped:
            result_x = get_draw_pos_impl(
                view,
                fg,
                p[0],
                bin_widths[0],
                fg.x_discrete_kind.discrete_type(),
                AxisKind.X,
            )
            result_y = get_draw_pos_impl(
                view,
                fg,
                cur_stack,
                bin_widths[1],
                fg.y_discrete_kind.discrete_type(),
                AxisKind.Y,
            )
        else:
            result_x = get_draw_pos_impl(
                view,
                fg,
                cur_stack,
                bin_widths[0],
                fg.x_discrete_kind.discrete_type(),
                AxisKind.X,
            )
            result_y = get_draw_pos_impl(
                view,
                fg,
                p[1],
                bin_widths[1],
                fg.y_discrete_kind.discrete_type(),
                AxisKind.Y,
            )
    else:
        raise GGException("not implemented yet")

    return Coord(x=result_x, y=result_y)


def draw_error_bar(
    view: ViewPort,
    fg: FilledGeomErrorBar,
    pos: Coord,
    df: pd.DataFrame,
    idx: int,
    style: Style,
) -> GOComposite:
    x_min, x_max, y_min, y_max = read_error_data(df, idx, fg)

    if x_min is not None or x_max is not None:
        if view.x_scale is None:
            raise GGException("exected view.x_scale")
        if style.error_bar_kind is None:
            raise GGException("exepcted error bar")

        error_up = Coord1D.create_data(x_max or pos.x.pos, view.x_scale, AxisKind.X)
        error_down = Coord1D.create_data(x_min or pos.x.pos, view.x_scale, AxisKind.X)
        data = InitErrorBarData(
            view=view,
            point=pos,
            error_up=error_up,
            error_down=error_down,
            axis_kind=AxisKind.X,
            error_bar_kind=style.error_bar_kind,
            style=style,
        )
        result = init_error_bar(data)
        return result
    if y_min is not None or y_max is not None:
        if view.y_scale is None:
            raise GGException("exected view.x_scale")
        if style.error_bar_kind is None:
            raise GGException("exepcted error bar")

        error_up = Coord1D.create_data(y_max or pos.y.pos, view.y_scale, AxisKind.Y)
        error_down = Coord1D.create_data(y_min or pos.y.pos, view.y_scale, AxisKind.Y)

        data = InitErrorBarData(
            view=view,
            point=pos,
            error_up=error_up,
            error_down=error_down,
            axis_kind=AxisKind.X,
            error_bar_kind=style.error_bar_kind,
            style=style,
        )
        result = init_error_bar(data)
        return result

    raise GGException("expected x_min or x_max or y_min or y_max")


def draw_raster(
    view: ViewPort, fg: FilledGeom, fg_raster: FilledGeomRaster, df: pd.DataFrame
):
    """
    TODO high priority -> FilledGeomRaster should be either
    a subclass of FilledGeom or included inside
    """
    max_x_col = fg.x_scale.high
    min_x_col = fg.x_scale.low
    max_y_col = fg.y_scale.high
    min_y_col = fg.y_scale.low
    wv, hv = read_width_height(df, 0, fg)

    height = max_y_col - min_y_col + hv
    width = max_x_col - min_x_col + wv

    num_x = round(width / wv)
    num_y = round(height / hv)
    c_map = fg_raster.data.color_scale

    def draw_callback():
        result = np.zeros(len(df), dtype=np.uint32)
        x_t = df[fg.x_col].to_numpy(dtype=float)
        y_t = df[fg.y_col].to_numpy(dtype=float)
        z_t = df[fg.fill_col].to_numpy(dtype=float)
        z_scale = fg.fill_data_scale

        for idx in range(len(df)):
            x = round((x_t[idx] - min_x_col) / wv)
            y = round((y_t[idx] - min_y_col) / hv)
            colors_high = len(c_map.colors) - 1

            color_idx = round(
                colors_high * ((z_t[idx] - z_scale.low) / (z_scale.high - z_scale.low))
            )
            color_idx = max(0, min(colors_high, color_idx))
            c_val = c_map.colors[color_idx]

            result[((num_y - y - 1) * num_x) + x] = c_val

        return result

    def data_c1(at: float, ax: AxisKind, view) -> Coord1D:
        scale = view.x_scale if ax == AxisKind.X else view.y_scale
        return Coord1D.create_data(pos=at, scale=scale, axis_kind=ax)

    def c_data(px: float, py: float, view) -> Coord:
        return Coord(x=data_c1(px, AxisKind.X, view), y=data_c1(py, AxisKind.Y, view))

    data = InitRasterData(callback=draw_callback, num_x=num_x, num_y=num_y)
    raster = init_raster(
        view=view,
        origin=c_data(min_x_col, max_y_col + hv, view),
        width=DataUnit(width),
        height=DataUnit(height),
        init_raster_data=data,
    )
    view.add_obj(raster)


def draw(
    view: ViewPort,
    fg: FilledGeom,
    pos: Coord,
    y,
    bin_widths: Tuple[float, float],
    df: pd.DataFrame,
    idx: int,
    style: Style,
):
    fg.geom.kind.draw(
        view,
        fg,
        pos,
        y,
        bin_widths,
        df,
        idx,
        style,
    )
