from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from python_ggplot.core.coord.objects import Coord, Coord1D, RelativeCoordType
from python_ggplot.core.objects import AxisKind, GGException, Style
from python_ggplot.core.units.objects import DataUnit, Quantity, RelativeUnit, UnitType
from python_ggplot.datamancer_pandas_compat import GGValue, VNull
from python_ggplot.gg_geom import FilledGeomDiscrete  # Geom,
from python_ggplot.gg_geom import (
    FilledGeom,
    FilledGeomErrorBar,
    FilledGeomHistogram,
    FilledGeomRaster,
    FilledGeomTitle,
    GeomType,
    HistogramDrawingStyle,
)
from python_ggplot.gg_styles import GGStyle, merge_user_style
from python_ggplot.gg_types import PREV_VALS_COL  # OutsideRangeKind,
from python_ggplot.gg_types import BinPositionType, DiscreteType, PositionType, Theme
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


def is_num(x: Any):
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


def get_xy(x_t: Any, y_t: Any, i: Any):
    x = 0.0 if pd.isna(x_t[i]) else x_t[i]
    y = 0.0 if pd.isna(y_t[i]) else y_t[i]

    # there was a check using %~ which i think it ensure that
    # we dont have float32 and float64, so the rest of the logic may not be needed

    return (x, y)


def read_or_calc_bin_width(
    df: pd.DataFrame,
    idx: int,
    data_col: str,
    dc_kind: DiscreteType,
    col: str = "binWidths",
) -> float:
    # TODO clean up later
    if dc_kind == DiscreteType.CONTINUOUS:
        if col in df.columns:
            if pd.isna(df.iloc[idx][col]):  # type: ignore
                return 0.0
            return df.iloc[idx][col]  # type: ignore
        elif idx < len(df) - 1:
            high_val = float(df.iloc[idx + 1][data_col])  # tpye: ignore
            if pd.isna(high_val):
                if idx <= 0:
                    raise GGException("expected idx> 0")
                return df.iloc[idx][data_col] - df.iloc[idx - 1][data_col]  # type: ignore
            else:
                return high_val - df.iloc[idx][data_col]  # type: ignore
    elif dc_kind == DiscreteType.DISCRETE:
        return 0.8

    raise GGException()


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
    fg: FilledGeom,
    geom: Union[FilledGeomTitle, FilledGeomRaster],
) -> Tuple[float, float]:
    """
    Todo high priority
        geom: Union[FilledGeomTitle, FilledGeomRaster] is a temp fix
        i think we need filled_geom.geom = FilledGeomTitle|FilledGeomRaster
        this fucntionality is mostly done, but needs some rewiring
    """
    width = 1.0
    height = 1.0

    # TODO deal with pyright iloc types down the line
    # this is more of a global issue, other priorties for now
    if geom.data.width is not None:
        width = float(df[geom.data.width].iloc[idx])  # type: ignore

    if geom.data.height is not None:
        height = float(df[geom.data.height].iloc[idx])  # type: ignore

    return width, height


def read_text(df: pd.DataFrame, idx: int, fg: FilledGeom) -> str:
    return df[fg.text].iloc[idx]  # type: ignore


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
            y = fg.x_discrete_kind.get_label_seq()[i]
            result[(x, y)] = i + 1
    else:
        for i in range(rows):
            y = fg.y_discrete_kind.get_label_seq()[i]
            for j in range(cols):
                x = fg.get_x_label_seq()[j]
                result[(x, y)] = (i + 1) * (cols + 2) + (j + 1)

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
    p: Tuple[float, float],
    bin_widths: Tuple[float, float],
    df: pd.DataFrame,
    idx: int,
) -> Coord:

    coords_flipped = False

    geom_type = fg.geom_type
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

        result_x: Coord1D = get_draw_pos_impl(
            view,
            fg,
            mp[0],
            bin_widths[0],
            fg.x_discrete_kind.discrete_type,
            AxisKind.X,
        )
        result_y: Coord1D = get_draw_pos_impl(
            view,
            fg,
            mp[1],
            bin_widths[1],
            fg.y_discrete_kind.discrete_type,
            AxisKind.Y,
        )
        return Coord(x=result_x, y=result_y)

    elif position == PositionType.STACK:
        if not (
            (
                fg.geom.kind.geom_type == GeomType.HISTOGRAM
                and histogram_drawing_style == HistogramDrawingStyle.BARS
            )
            or fg.geom.kind.geom_type == GeomType.BAR
        ):
            cur_stack = p[1]
        else:
            cur_stack = df[PREV_VALS_COL].iloc[idx]  # type: ignore

        if not coords_flipped:
            result_x = get_draw_pos_impl(
                view,
                fg,
                p[0],
                bin_widths[0],
                fg.x_discrete_kind.discrete_type,
                AxisKind.X,
            )
            result_y = get_draw_pos_impl(
                view,
                fg,
                cur_stack,
                bin_widths[1],
                fg.y_discrete_kind.discrete_type,
                AxisKind.Y,
            )
            return Coord(x=result_x, y=result_y)
        else:
            result_x = get_draw_pos_impl(
                view,
                fg,
                cur_stack,
                bin_widths[0],
                fg.x_discrete_kind.discrete_type,
                AxisKind.X,
            )
            result_y = get_draw_pos_impl(
                view,
                fg,
                p[1],
                bin_widths[1],
                fg.y_discrete_kind.discrete_type,
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
    # TODO high priority this takes an additional parameter,
    # we have to ensure filled geom is of type raster and it has the right data
    wv, hv = read_width_height(df, 0, fg)

    height = max_y_col - min_y_col + hv
    width = max_x_col - min_x_col + wv

    num_x = round(width / wv)
    num_y = round(height / hv)
    c_map = fg_raster.data.color_scale

    def draw_callback():
        # TODO this needs fixing, fine for now
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


def calc_bin_widths(df: pd.DataFrame, idx: int, fg: FilledGeom) -> Tuple[float, float]:
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
                df, idx, fg.x_col, dc_kind=fg.discrete_type_x
            )
        else:
            y_width = read_or_calc_bin_width(
                df, idx, fg.y_col, dc_kind=fg.discrete_type_y
            )

    elif geom_type in [GeomType.TILE, GeomType.RASTER]:
        # TODO this needs to pass the tile/raster data
        # once the objects are restructured we can take in the right type
        x_width, y_width = read_width_height(df, idx, fg)

    return x_width, y_width


def move_bin_positions(
    point: Tuple[float, float], bin_widths: tuple[float, float], fg: FilledGeom
):
    if fg.geom.bin_position is None:
        raise GGException("expected a bin position")

    coord_flipped = False
    x, y = point
    bin_width_x, bin_width_y = bin_widths

    if fg.geom.kind.geom_type == GeomType.TILE:
        x = move_bin_position(x, fg.geom.bin_position, bin_width_x)
        y = move_bin_position(y, fg.geom.bin_position, bin_width_y)
        return x, y
    else:
        if coord_flipped is False:
            x = move_bin_position(x, fg.geom.bin_position, bin_width_x)
        else:
            y = move_bin_position(y, fg.geom.bin_position, bin_width_y)
        return x, y


def get_view(
    view_map: Dict[Any, Any], point: Tuple[float, float], fg: FilledGeom
) -> int:
    px = point[0] if fg.is_discrete_x() else None
    py = point[1] if fg.is_discrete_y() else None
    return view_map[(px, py)]


def extend_line_to_axis(
    line_points: List[Coord],
    ax_kind: AxisKind,
    df: pd.DataFrame,
    filled_geom: FilledGeom,
) -> List[Coord]:
    """
    TODO medium/low priority clean up this logic, after geom re structrure is done
    """
    l_start: Coord = line_points[0]
    l_end: Coord = line_points[-1]

    geom_type: GeomType = filled_geom.geom_type
    discrete_type_x: DiscreteType = filled_geom.discrete_type_x
    discrete_type_y: DiscreteType = filled_geom.discrete_type_y

    if ax_kind == AxisKind.X:
        l_start.y.pos = 0.0
        if geom_type == GeomType.FREQ_POLY:
            bin_width = read_or_calc_bin_width(
                df, 0, filled_geom.x_col, dc_kind=discrete_type_x
            )
            l_start.x.pos = l_start.x.pos - bin_width

        line_points.insert(0, l_start)

        l_end.y.pos = 0.0
        if geom_type == GeomType.FREQ_POLY:
            bin_width = read_or_calc_bin_width(
                df, len(df) - 2, filled_geom.x_col, dc_kind=discrete_type_x
            )
            l_end.x.pos = l_end.x.pos + bin_width

        line_points.append(l_end)

    elif ax_kind == AxisKind.Y:
        l_start.x.pos = 0.0
        if geom_type == GeomType.FREQ_POLY:
            bin_width = read_or_calc_bin_width(
                df, 0, filled_geom.y_col, dc_kind=discrete_type_y
            )
            l_start.y.pos = l_start.y.pos - bin_width

        line_points.insert(0, l_start)

        l_end.x.pos = 0.0
        if geom_type == GeomType.FREQ_POLY:
            bin_width = read_or_calc_bin_width(
                df, len(df) - 2, filled_geom.y_col, dc_kind=discrete_type_y
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
        df, 0, filled_geom.x_col, dc_kind=discrete_type_x
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
            df, 0, filled_geom.x_col, dc_kind=discrete_type_x
        )
        cur_p = line_points[idx]

        cur_y = cur_p.y.pos
        point.y.pos = cur_y
        result.append(point)

        cur_x = cur_x + bin_width
        point.x.pos = cur_x
        result.append(point)

    return result


def draw_sub_df(
    view: ViewPort,
    fg: FilledGeom,
    view_map: Dict[Tuple[Any, Any], int],
    df: pd.DataFrame,
    styles: List[GGStyle],
    theme: Theme,
) -> None:
    """
    TODO restructure this down the line
    """
    # this was used in get_x_y, we may need that soon
    # x_outside_range = theme.x_outside_range or OutsideRangeKind.CLIP
    # y_outside_range = theme.y_outside_range or OutsideRangeKind.CLIP
    bin_widths: Tuple[float, float] = tuple()
    geom_type = fg.geom_type

    style = merge_user_style(styles[0], fg)
    loc_view: ViewPort = view
    view_idx = 0

    need_bin_width = geom_type in {
        GeomType.BAR,
        GeomType.HISTOGRAM,
        GeomType.TILE,
        GeomType.RASTER,
    } or fg.geom.bin_position in {BinPositionType.CENTER, BinPositionType.RIGHT}

    line_points: List[Coord] = []
    if geom_type not in {GeomType.RASTER}:
        x_tensor = df[fg.x_col]  # type: ignore
        y_tensor = df[fg.y_col]  # type: ignore

        last_element: int = len(df) - 2
        if fg.geom.bin_position == BinPositionType.NONE:
            last_element = len(df) - 1

        for i in range(last_element + 1):
            if len(styles) > 1:
                style = merge_user_style(styles[i], fg)

            # TODO high priority: double check this logic
            # the origin get_xy has a lot of logic, that seemed redundant
            # this has a high chance of introducing a bug
            point = get_xy(
                x_tensor,
                y_tensor,
                i,
            )

            if view_map:
                view_idx = get_view(view_map, point, fg)
                loc_view = view.children[view_idx]

            if need_bin_width:
                bin_widths = calc_bin_widths(df, i, fg)
                move_bin_positions(point, bin_widths, fg)

            pos = get_draw_pos(loc_view, view_idx, fg, point, bin_widths, df, i)

            if fg.geom.position in {PositionType.IDENTITY, PositionType.STACK}:
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
                view[view_idx] = loc_view

    if not view_map:
        view = loc_view

    if geom_type == GeomType.HISTOGRAM and fg.get_histogram_draw_style():
        return
    elif geom_type == GeomType.HISTOGRAM:
        line_points = convert_points_to_histogram(df, fg, line_points)

    if geom_type in {GeomType.LINE, GeomType.FREQ_POLY, GeomType.HISTOGRAM}:
        if len(styles) == 1:
            style = merge_user_style(styles[0], fg)
            if style.fill_color is None:
                raise GGException("expected fill color")
            if style.fill_color.a == 0.0 or geom_type == GeomType.FREQ_POLY:
                line_points = extend_line_to_axis(line_points, AxisKind.X, df, fg)
            poly_line = init_poly_line_from_points(
                view, [i.point() for i in line_points], style
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
                    view, [line_points[i].point(), line_points[i + 1].point()], style
                )
                view.add_obj(poly_line)
    elif geom_type == GeomType.RASTER:
        # TODO: currently ignores the line_points completely
        # TODO high priority fix GEOM object structure
        draw_raster(view, fg, cast(FilledGeomRaster, fg), df)


def create_gobj_from_geom(
    view: ViewPort, fg: FilledGeom, theme: Theme, label_val: Optional[Any] = None
):
    prepare_views(view, fg, theme)
    view_map = calc_view_map(fg)

    for lab, _, styles, sub_df in enumerate_data(fg):
        if label_val is not None:
            if label_val not in lab:
                # skip this label
                continue
        draw_sub_df(view, fg, view_map, sub_df, styles, theme)
