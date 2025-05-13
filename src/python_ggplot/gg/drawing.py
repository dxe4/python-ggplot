from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast, no_type_check

import numpy as np
import pandas as pd

from python_ggplot.core.coord.objects import Coord, Coord1D, RelativeCoordType
from python_ggplot.core.objects import AxisKind, GGException, Style
from python_ggplot.core.units.objects import DataUnit, Quantity, UnitType
from python_ggplot.gg.datamancer_pandas_compat import VNull
from python_ggplot.gg.geom.base import (
    FilledGeom,
    FilledGeomDiscrete,
    FilledGeomErrorBar,
    FilledGeomHistogram,
    FilledGeomRaster,
    FilledGeomTitle,
    GeomType,
    HistogramDrawingStyle,
)
from python_ggplot.gg.styles.utils import merge_user_style
from python_ggplot.gg.types import (
    PREV_VALS_COL,
    BinPositionType,
    DiscreteType,
    GGStyle,
    OutsideRangeKind,
    PositionType,
    Theme,
)
from python_ggplot.graphics.draw import layout
from python_ggplot.graphics.initialize import (
    InitErrorBarData,
    InitRasterData,
    init_coord_1d,
    init_error_bar,
    init_poly_line_from_points,
    init_raster,
)
from python_ggplot.graphics.objects import GOComposite
from python_ggplot.graphics.views import ViewPort


class GetXYContinueException(GGException):
    """
    TODO this is a bit of a hack
    the original code uses a macro
    and it injects a continue statement in the main loop
    we can re-wire this logic later
    but for now this is the easiest way of doing that
    """


@dataclass
class GetXY:
    x_series: pd.Series  # type: ignore
    y_series: pd.Series  # type: ignore
    x_outside_range: OutsideRangeKind
    y_outside_range: OutsideRangeKind
    filled_geom: FilledGeom
    idx: int
    theme: Theme
    view: ViewPort
    # TODO CRITICAL implement
    x_maybe_string: bool = True

    def _change_if_needed(
        self, outside_range: OutsideRangeKind, value: Any, potential_value: Any
    ):
        if outside_range == OutsideRangeKind.DROP:
            raise GetXYContinueException("skip the loop")
        elif outside_range == OutsideRangeKind.CLIP:
            return potential_value
        elif outside_range == OutsideRangeKind.NONE:
            return value
        else:
            raise GGException("unexpected outside kind range")

    def calculate(self) -> Tuple[float, float, Any]:
        """
        TODO this assumes view.x_scale and view.y_scale
        need to decide how to handle that,
        defualt to -inf and +inf, raise exception, or do nothing?
        """
        x_is_str = isinstance(self.x_series.iloc[self.idx], str)  # type: ignore
        if x_is_str:
            # TODO is this correct?
            x = 0.0
        else:
            if pd.isna(self.x_series.iloc[self.idx]):  # type: ignore
                x = 0.0
            else:
                x = float(self.x_series.iloc[self.idx])  # type: ignore

        if pd.isna(self.y_series.iloc[self.idx]):  # type: ignore
            y = 0.0
        else:
            temp = self.y_series.iloc[self.idx]
            if isinstance(temp, str):
                # TODO CRITICAL+ double check this, how is y str handled
                y = temp
            else:
                y = float(self.y_series.iloc[self.idx])  # type: ignore

        # TODO CRITICAL, easy task
        # write is_continuous and use that
        # although this is binary discrete/continuous
        # the reality is nothing prevents it from having a third dimension
        # and this can make hard to reproduce bugs
        # an example can be a hybrid case
        # discrete in a range continuous in another
        if not self.filled_geom.is_discrete_x():
            if x > self.view.x_scale.high:
                x = self._change_if_needed(
                    self.x_outside_range, x, self.theme.x_margin_range.high
                )
            if x < self.view.x_scale.low:
                x = self._change_if_needed(
                    self.x_outside_range, x, self.theme.x_margin_range.low
                )

        if not self.filled_geom.is_discrete_y():
            if y > self.view.y_scale.high:
                y = self._change_if_needed(
                    self.y_outside_range, y, self.theme.y_margin_range.high
                )
            if y < self.view.y_scale.low:
                y = self._change_if_needed(
                    self.y_outside_range, y, self.theme.y_margin_range.low
                )

        if x_is_str:
            return (x, y, self.x_series.iloc[self.idx])  # type: ignore
        else:
            return (x, y, x)  # type: ignore


@no_type_check
def _continuous_bin_width(
    df: pd.DataFrame, idx: int, column: str, data_col: str
) -> float:
    if column in df.columns:
        if pd.isna(df.iloc[idx][column]):
            return 0.0
        return df.iloc[idx][column]
    elif idx < len(df) - 1:
        high_val = float(df.iloc[idx + 1][data_col])
        if pd.isna(high_val):
            if idx <= 0:
                raise GGException("expected idx> 0")
            return df.iloc[idx][data_col] - df.iloc[idx - 1][data_col]
        else:
            return high_val - df.iloc[idx][data_col]


def read_or_calc_bin_width(
    df: pd.DataFrame,
    idx: int,
    data_col: str,
    dc_kind: DiscreteType,
    col: str = "binWidths",
) -> float:
    if dc_kind == DiscreteType.CONTINUOUS:
        return _continuous_bin_width(df, idx, col, data_col)
    elif dc_kind == DiscreteType.DISCRETE:
        return 0.8
    else:
        raise GGException("Discrete type should be DISCRETE or CONTINUOUS")


def move_bin_position(x: float, bp_kind: BinPositionType, bin_width: float):
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
        result["x_min"] = float(df[fg.x_min].iloc[idx])  # type: ignore
    if fg.x_max is not None:
        result["x_max"] = float(df[fg.x_max].iloc[idx])  # type: ignore
    if fg.y_min is not None:
        result["y_min"] = float(df[fg.y_min].iloc[idx])  # type: ignore
    if fg.y_max is not None:
        result["y_max"] = float(df[fg.y_max].iloc[idx])  # type: ignore

    return result["x_min"], result["x_max"], result["y_min"], result["y_max"]


def read_width_height(
    df: pd.DataFrame,
    idx: int,
    fg: Union[FilledGeomTitle, FilledGeomRaster],
) -> Tuple[float, float]:
    # TODO read_width_height can go on the TITLE / RASTER mixin
    width = 1.0
    height = 1.0

    # TODO deal with pyright iloc types down the line
    # this is more of a global issue, other priorties for now
    if fg.data.width is not None:
        width = float(df[fg.data.width].iloc[idx])  # type: ignore

    if fg.data.height is not None:
        height = float(df[fg.data.height].iloc[idx])  # type: ignore

    return width, height


def read_text(df: pd.DataFrame, idx: int, fg: FilledGeom) -> str:
    return df[fg.text].iloc[idx]  # type: ignore


def get_cols_and_rows(fg: FilledGeom) -> Tuple[int, int]:
    cols, rows = 1, 1

    if fg.is_discrete_x():
        f_geom_ = cast(FilledGeomDiscrete, fg.gg_data.x_discrete_kind)
        cols = len(f_geom_.label_seq)

    if fg.is_discrete_y():
        f_geom_ = cast(FilledGeomDiscrete, fg.gg_data.y_discrete_kind)
        rows = len(f_geom_.label_seq)

    return cols, rows


def prepare_views(view: ViewPort, fg: FilledGeom, theme: Theme):

    cols, rows = get_cols_and_rows(fg)

    discrete_margin = theme.discrete_scale_margin or Quantity.relative(0.0)

    widths: List[Quantity] = []
    heights: List[Quantity] = []

    if cols > 1:
        ind_widths: List[Quantity] = [Quantity.relative(0.0) for _ in range(cols)]
        cols += 2
        widths = [discrete_margin] + ind_widths + [discrete_margin]

    if rows > 1:
        ind_heights: List[Quantity] = [Quantity.relative(0.0) for _ in range(rows)]
        rows += 2
        heights = [discrete_margin] + ind_heights + [discrete_margin]

    layout(
        view,
        cols=cols,
        rows=rows,
        col_widths=widths,
        row_heights=heights,
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
            y = fg.gg_data.y_discrete_kind.get_label_seq()[i]
            result[(x, y)] = i + 1
    else:
        for i in range(rows):
            # TODO CRITICAL+
            # there is an issue with label seq here
            # we can by pass it for now to fix the rest of the issues
            # the logic here is most likely fine,
            # but the issue caused by wrong label seq before coming here
            if i >= len(fg.gg_data.y_discrete_kind.get_label_seq()):
                break
            y = fg.gg_data.y_discrete_kind.get_label_seq()[i]
            for j in range(cols):
                x = fg.get_x_label_seq()[j]
                result[(x, y)] = (i + 1) * (cols + 2) + (j + 1)

    return result


def get_discrete_histogram(width: float, ax_kind: AxisKind) -> Coord1D:
    if ax_kind == AxisKind.X:
        left = (1.0 - width) / 2.0

        return init_coord_1d(left, AxisKind.X, UnitType.RELATIVE)
    else:
        # TODO CRITICAL easy task
        # double check this.
        # templace c1 in ginger has default AxisKind.X which is used in this case
        # changed to the sensible Y but it still needs double chcking
        top = 1.0
        return init_coord_1d(top, AxisKind.Y, UnitType.RELATIVE)


def get_discrete_point() -> Coord1D:
    return RelativeCoordType(0.5)


def get_discrete_line(view: ViewPort, axis_kind: AxisKind) -> Coord1D:
    center_x, center_y = view.get_center()
    lookup = {
        AxisKind.X: init_coord_1d(center_x, AxisKind.X, UnitType.RELATIVE),
        # TODO sanity check this, original code is using AxisKind.X (inferred as default)
        # is this correct?
        # changed to the sensible Y but it still needs double chcking
        AxisKind.Y: init_coord_1d(center_y, AxisKind.Y, UnitType.RELATIVE),
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
    # TODO Value in source, change later
    val: Any,
    width: float,
    discrete_type: DiscreteType,
    ax_kind: AxisKind,
) -> Coord1D:

    fg_geom_type = fg.geom_type
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
    raise GGException("could not get point")


def get_draw_pos(
    view: ViewPort,
    view_idx: int,
    fg: FilledGeom,
    point: Tuple[float, float],
    bin_widths: Tuple[float, float],
    df: pd.DataFrame,
    idx: int,
) -> Coord:

    coords_flipped = False

    geom_type = fg.geom_type
    position = fg.gg_data.geom.gg_data.position

    # TODO sanity check if we need more types than hist
    histogram_drawing_style = getattr(fg, "histogram_drawing_style", None)

    if position == PositionType.IDENTITY:
        mp = deepcopy(list(point))
        if geom_type == GeomType.BAR or (
            geom_type == GeomType.HISTOGRAM
            and histogram_drawing_style == HistogramDrawingStyle.BARS
        ):
            if not coords_flipped:
                mp[1] = 0.0
            else:
                mp[0] = 0.0

        result_x: Coord1D = get_draw_pos_impl(
            view,
            fg,
            mp[0],
            bin_widths[0],
            fg.gg_data.x_discrete_kind.discrete_type,
            AxisKind.X,
        )
        result_y: Coord1D = get_draw_pos_impl(
            view,
            fg,
            mp[1],
            bin_widths[1],
            fg.gg_data.y_discrete_kind.discrete_type,
            AxisKind.Y,
        )
        return Coord(x=result_x, y=result_y)

    elif position == PositionType.STACK:
        if not (
            (
                fg.gg_data.geom.geom_type == GeomType.HISTOGRAM
                and histogram_drawing_style == HistogramDrawingStyle.BARS
            )
            or fg.gg_data.geom.geom_type == GeomType.BAR
        ):
            cur_stack = point[1]
        else:
            cur_stack = float(df[PREV_VALS_COL].iloc[idx])  # type: ignore

        if not coords_flipped:
            result_x = get_draw_pos_impl(
                view,
                fg,
                point[0],
                bin_widths[0],
                fg.gg_data.x_discrete_kind.discrete_type,
                AxisKind.X,
            )
            result_y = get_draw_pos_impl(
                view,
                fg,
                cur_stack,
                bin_widths[1],
                fg.gg_data.y_discrete_kind.discrete_type,
                AxisKind.Y,
            )

            return Coord(x=result_x, y=result_y)
        else:
            result_x = get_draw_pos_impl(
                view,
                fg,
                cur_stack,
                bin_widths[0],
                fg.gg_data.x_discrete_kind.discrete_type,
                AxisKind.X,
            )
            result_y = get_draw_pos_impl(
                view,
                fg,
                point[1],
                bin_widths[1],
                fg.gg_data.y_discrete_kind.discrete_type,
                AxisKind.Y,
            )
            return Coord(x=result_x, y=result_y)
    else:
        raise GGException("not implemented yet")


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
            axis_kind=AxisKind.Y,
            error_bar_kind=style.error_bar_kind,
            style=style,
        )
        result = init_error_bar(data)
        return result

    raise GGException("expected x_min or x_max or y_min or y_max")


def draw_raster(view: ViewPort, fg: FilledGeomRaster, df: pd.DataFrame):
    max_x_col = fg.gg_data.x_scale.high
    min_x_col = fg.gg_data.x_scale.low
    max_y_col = fg.gg_data.y_scale.high
    min_y_col = fg.gg_data.y_scale.low

    wv, hv = read_width_height(df, 0, fg)

    height = max_y_col - min_y_col + hv
    width = max_x_col - min_x_col + wv

    num_x = round(width / wv)
    num_y = round(height / hv)
    c_map = fg.data.color_scale

    def draw_callback():
        # TODO this needs fixing, fine for now
        result = np.zeros(len(df), dtype=np.uint32)
        x_t = df[fg.gg_data.x_col].to_numpy(dtype=float)  # type: ignore
        y_t = df[fg.gg_data.y_col].to_numpy(dtype=float)  # type: ignore
        z_t = df[fg.data.fill_col].to_numpy(dtype=float)  # type: ignore
        z_scale = fg.data.fill_data_scale

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

    def data_c1(at: float, ax: AxisKind, view: ViewPort) -> Coord1D:
        scale = view.x_scale if ax == AxisKind.X else view.y_scale
        if scale is None:
            raise GGException("expected scale")
        return Coord1D.create_data(pos=at, scale=scale, axis_kind=ax)

    def c_data(px: float, py: float, view: ViewPort) -> Coord:
        return Coord(x=data_c1(px, AxisKind.X, view), y=data_c1(py, AxisKind.Y, view))

    data = InitRasterData(
        callback=draw_callback,  # type: ignore TODO check this later
        num_x=num_x,
        num_y=num_y,
    )
    raster = init_raster(
        view=view,
        origin=c_data(min_x_col, max_y_col + hv, view),
        width=DataUnit(width),
        height=DataUnit(height),
        init_raster_data=data,
    )
    view.add_obj(raster)


def gg_draw(
    view: ViewPort,
    fg: FilledGeom,
    pos: Coord,
    # TODO Value in origin, this will be fixed at later stage
    y: Any,
    bin_widths: Tuple[float, float],
    df: pd.DataFrame,
    idx: int,
    style: Style,
):
    fg.gg_data.geom.draw_geom(
        view,
        fg,
        pos,
        y,
        bin_widths,
        df,
        idx,
        deepcopy(style),
    )


def calc_bin_widths(
    df: pd.DataFrame, idx: int, fg: FilledGeom  # type: ignore
) -> Tuple[float, float]:
    x_width: float = 0.0
    y_width: float = 0.0
    coord_flipped = False
    geom_type: GeomType = fg.geom_type

    if geom_type in [
        GeomType.HISTOGRAM,
        GeomType.BAR,
        GeomType.POINT,
        GeomType.LINE,
        GeomType.FREQ_POLY,
        GeomType.ERROR_BAR,
        GeomType.TEXT,
    ]:
        if not coord_flipped:
            x_width = read_or_calc_bin_width(
                df, idx, fg.gg_data.x_col, dc_kind=fg.discrete_type_x
            )
        else:
            y_width = read_or_calc_bin_width(
                df, idx, fg.gg_data.y_col, dc_kind=fg.discrete_type_y
            )

    elif geom_type in [GeomType.TILE, GeomType.RASTER]:
        x_width, y_width = read_width_height(df, idx, fg)  # type: ignore

    return x_width, y_width


def move_bin_positions(
    point: Tuple[float, float], bin_widths: tuple[float, float], fg: FilledGeom
):
    if fg.gg_data.geom.gg_data.bin_position is None:
        raise GGException("expected a bin position")

    coord_flipped = False
    x, y = point
    bin_width_x, bin_width_y = bin_widths
    if fg.gg_data.geom.geom_type == GeomType.TILE:
        x = move_bin_position(x, fg.gg_data.geom.gg_data.bin_position, bin_width_x)
        y = move_bin_position(y, fg.gg_data.geom.gg_data.bin_position, bin_width_y)
        return x, y
    else:
        if coord_flipped is False:
            x = move_bin_position(x, fg.gg_data.geom.gg_data.bin_position, bin_width_x)
        else:
            y = move_bin_position(y, fg.gg_data.geom.gg_data.bin_position, bin_width_y)
        return x, y


def get_view(view_map: Dict[Any, Any], point: Tuple[Any, Any], fg: FilledGeom) -> int:
    # TODO Vnull has to be deleted one day, or maybe not?
    px = point[0] if fg.is_discrete_x() else VNull()
    py = point[1] if fg.is_discrete_y() else VNull()
    return view_map[(px, py)]


def extend_line_to_axis(
    line_points: List[Coord],
    ax_kind: AxisKind,
    df: pd.DataFrame,  # type: ignore
    filled_geom: FilledGeom,
) -> List[Coord]:
    """
    TODO medium/low priority clean up this logic, after geom re structrure is done
    """
    l_start: Coord = deepcopy(line_points[0])
    l_end: Coord = deepcopy(line_points[-1])

    geom_type: GeomType = filled_geom.geom_type
    discrete_type_x: DiscreteType = filled_geom.discrete_type_x
    discrete_type_y: DiscreteType = filled_geom.discrete_type_y

    if ax_kind == AxisKind.X:
        l_start.y.pos = 0.0
        if geom_type == GeomType.FREQ_POLY:
            bin_width = read_or_calc_bin_width(
                df, 0, filled_geom.gg_data.x_col, dc_kind=discrete_type_x
            )
            l_start.x.pos = l_start.x.pos - bin_width

        line_points.insert(0, l_start)

        l_end.y.pos = 0.0
        if geom_type == GeomType.FREQ_POLY:
            bin_width = read_or_calc_bin_width(
                df, len(df) - 2, filled_geom.gg_data.x_col, dc_kind=discrete_type_x
            )
            l_end.x.pos = l_end.x.pos + bin_width

        line_points.append(l_end)

    elif ax_kind == AxisKind.Y:
        l_start.x.pos = 0.0
        if geom_type == GeomType.FREQ_POLY:
            bin_width = read_or_calc_bin_width(
                df, 0, filled_geom.gg_data.y_col, dc_kind=discrete_type_y
            )
            l_start.y.pos = l_start.y.pos - bin_width

        line_points.insert(0, l_start)

        l_end.x.pos = 0.0
        if geom_type == GeomType.FREQ_POLY:
            bin_width = read_or_calc_bin_width(
                df, len(df) - 2, filled_geom.gg_data.y_col, dc_kind=discrete_type_y
            )
            l_end.y.pos = l_end.y.pos + bin_width

        line_points.append(l_end)

    return line_points


def convert_points_to_histogram(
    df: pd.DataFrame, filled_geom: FilledGeom, line_points: List[Coord]
) -> List[Coord]:
    """
    TODO sanity check / unit test:
        is it safe to mutate the points or do we need to clone them?
        unclear at this stage
    """
    result: List[Coord] = []
    point = line_points[0]
    cur_x: float = point.x.pos
    cur_y: float = 0.0

    discrete_type_x: DiscreteType = filled_geom.discrete_type_x

    bin_width = read_or_calc_bin_width(
        df, 0, filled_geom.gg_data.x_col, dc_kind=discrete_type_x
    )

    point.x.pos = cur_x
    point.y.pos = cur_y
    result.append(point)

    cur_y = line_points[0].y.pos
    point.y.pos = cur_y
    result.append(point)

    cur_x = cur_x + bin_width
    point.x.pos = cur_x
    result.append(point)

    for idx in range(1, len(line_points)):
        bin_width = read_or_calc_bin_width(
            df, 0, filled_geom.gg_data.x_col, dc_kind=discrete_type_x
        )
        cur_p = line_points[idx]

        cur_y = cur_p.y.pos
        point.y.pos = cur_y
        result.append(point)

        cur_x = cur_x + bin_width
        point.x.pos = cur_x
        result.append(point)

    return result


def _needs_bin_width(geom_type: GeomType, bin_position: Optional[BinPositionType]):
    if geom_type in {
        GeomType.BAR,
        GeomType.HISTOGRAM,
        GeomType.TILE,
        GeomType.RASTER,
    }:
        return True
    if bin_position in {
        BinPositionType.CENTER,
        BinPositionType.RIGHT,
    }:
        return True
    return False


def draw_sub_df(
    view: ViewPort,
    fg: FilledGeom,
    view_map: Dict[Tuple[Any, Any], int],
    df: pd.DataFrame,  # type: ignore
    styles: List[GGStyle],
    theme: Theme,
):
    """
    TODO restructure this down the line
    """
    x_outside_range = theme.x_outside_range or OutsideRangeKind.CLIP
    y_outside_range = theme.y_outside_range or OutsideRangeKind.CLIP
    bin_widths: Tuple[float, float] = tuple()
    geom_type = fg.geom_type
    style = merge_user_style(styles[0], fg)
    loc_view: ViewPort = view
    view_idx = 0

    need_bin_width = _needs_bin_width(geom_type, fg.gg_data.geom.gg_data.bin_position)

    line_points: List[Coord] = []
    if geom_type != GeomType.RASTER:
        x_tensor = df[fg.gg_data.x_col]  # type: ignore
        y_tensor = df[fg.gg_data.y_col]  # type: ignore

        last_element: int = len(df) - 1
        if fg.gg_data.geom.gg_data.bin_position == BinPositionType.NONE:
            last_element = len(df)

        for i in range(last_element):
            if len(styles) > 1:
                style = merge_user_style(styles[i], fg)

            get_xy_obj = GetXY(
                x_series=x_tensor,  # type: ignore
                y_series=y_tensor,  # type: ignore
                x_outside_range=x_outside_range,
                y_outside_range=y_outside_range,
                filled_geom=fg,
                idx=i,
                theme=theme,
                view=view,
            )
            try:
                x_, y_, x_name = get_xy_obj.calculate()
                point = (x_, y_)
            except GetXYContinueException:
                continue

            if view_map:
                view_idx = get_view(view_map, (x_name, y_), fg)
                loc_view = view.children[view_idx]

            if need_bin_width:
                bin_widths = calc_bin_widths(df, i, fg)
                point = move_bin_positions(point, bin_widths, fg)
            else:
                bin_widths = (0.0, 0.0)

            pos = get_draw_pos(loc_view, view_idx, fg, point, bin_widths, df, i)
            if fg.gg_data.geom.gg_data.position in {
                PositionType.IDENTITY,
                PositionType.STACK,
            }:
                if geom_type in {
                    GeomType.LINE,
                    GeomType.FREQ_POLY,
                    GeomType.RASTER,
                }:
                    line_points.append(pos)
                elif geom_type == GeomType.HISTOGRAM:
                    temp = cast(FilledGeomHistogram, fg)
                    if temp.histogram_drawing_style == HistogramDrawingStyle.OUTLINE:
                        line_points.append(pos)
                    else:
                        if pos is None:
                            raise GGException("pos shouldnt be none")
                        gg_draw(loc_view, fg, pos, point[1], bin_widths, df, i, style)
                else:
                    if pos is None:
                        raise GGException("pos shouldnt be none")
                    gg_draw(loc_view, fg, pos, point[1], bin_widths, df, i, style)

            if view_map:
                view.children[view_idx] = loc_view

    if not view_map:
        view = loc_view

    if geom_type == GeomType.HISTOGRAM and fg.get_histogram_draw_style():
        return
    elif geom_type == GeomType.HISTOGRAM:
        line_points = convert_points_to_histogram(df, fg, line_points)

    if geom_type in {GeomType.LINE, GeomType.FREQ_POLY, GeomType.HISTOGRAM}:
        if len(styles) == 1:
            style = merge_user_style(deepcopy(styles[0]), fg)
            if style.fill_color is None:
                raise GGException("expected fill color")

            # TODO CRITICAL+ easy fix
            # this logic is correct but is triggered at the wrong conditions
            # Should not be triggered for geom_line (as draws a line from axis to point)
            # if style.fill_color.a == 0.0 or geom_type == GeomType.FREQ_POLY:
            if geom_type == GeomType.FREQ_POLY:
                line_points = extend_line_to_axis(line_points, AxisKind.X, df, fg)

            poly_line = init_poly_line_from_points(
                view, [i.point() for i in line_points], deepcopy(style)
            )
            view.add_obj(poly_line)
        else:
            # Since we don't support gradients on lines, we just draw from
            # (x1/y1) to (x2/y2) with the style of (x1/x2)
            print("WARNING: using non-gradient drawing of line with multiple colors!")
            if style.fill_color is None:
                raise GGException("expected fill color")

            if style.fill_color.a == 0.0 or geom_type == GeomType.FREQ_POLY:
                line_points = extend_line_to_axis(line_points, AxisKind.X, df, fg)
            for i in range(len(styles) - 1):
                style = merge_user_style(styles[i], fg)
                poly_line = init_poly_line_from_points(
                    view,
                    [line_points[i].point(), line_points[i + 1].point()],
                    deepcopy(style),
                )
                view.add_obj(poly_line)
    elif geom_type == GeomType.RASTER:
        draw_raster(view, cast(FilledGeomRaster, fg), df)


def create_gobj_from_geom(
    view: ViewPort, fg: FilledGeom, theme: Theme, label_val: Optional[Any] = None
):
    prepare_views(view, fg, theme)

    view_map = calc_view_map(fg)

    for lab, _, styles, sub_df in fg.enumerate_data():
        if label_val is not None and label_val not in lab:
            # skip this label
            continue
        draw_sub_df(view, fg, view_map, sub_df, styles, theme)
