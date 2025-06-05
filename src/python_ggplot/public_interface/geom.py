from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Set, Union

import pandas as pd

from python_ggplot.common.enum_literals import (
    BIN_BY_VALUES,
    BIN_POSITION_VALUES,
    ERROR_BAR_KIND_VALUES,
    HISTOGRAM_DRAWING_STYLE_VALUES,
    LINE_TYPE_VALUES,
    POSITION_VALUES,
    SMOOTH_METHOD_TYPE_VALUES,
    STAT_TYPE_VALUES,
)
from python_ggplot.core.objects import ErrorBarKind, GGException, LineType
from python_ggplot.core.units.objects import Quantity
from python_ggplot.gg.geom.base import (
    Geom,
    GeomABLine,
    GeomArea,
    GeomBar,
    GeomData,
    GeomErrorBar,
    GeomFreqPoly,
    GeomHistogram,
    GeomHLine,
    GeomLine,
    GeomPoint,
    GeomRaster,
    GeomRect,
    GeomText,
    GeomTile,
    GeomVLine,
    HistogramDrawingStyle,
)
from python_ggplot.gg.types import (
    Aesthetics,
    BinByType,
    BinPositionType,
    GgPlot,
    PositionType,
    PossibleColor,
    PossibleFloat,
    PossibleMarker,
    SmoothMethodType,
    StatKind,
    StatSmooth,
    StatType,
    Theme,
)
from python_ggplot.gg.utils import assign_identity_scales_get_style

id_counter = 1


def increment_id():
    global id_counter
    id_counter += 1


def get_gid():
    global id_counter
    id_counter += 1
    return id_counter - 1


CallableNoType = Callable[..., Any]


def fill_ids(aes: Aesthetics, gids: Set[int]) -> Aesthetics:
    result = deepcopy(aes)

    for field, _ in result.__dataclass_fields__.items():
        value = getattr(result, field)
        if value is not None:
            value = deepcopy(value)
            value.gg_data.ids = gids
            setattr(result, field, value)

    return result


def ggplot(data: pd.DataFrame, aes: Optional[Aesthetics] = None) -> GgPlot:
    # TODO CRITICAL, easy task
    # do we need to copy ?
    shallow_copy: pd.DataFrame = data.copy(deep=False)
    if aes is None:
        aes = Aesthetics()

    aes = fill_ids(aes, set(range(0, 65536)))

    result = GgPlot(
        data=shallow_copy,
        aes=aes,
        theme=Theme(discrete_scale_margin=Quantity.centimeters(0.2)),
    )
    return result


def geom_point(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    marker: PossibleMarker = None,
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = -1,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
) -> "Geom":

    if breaks is None:
        breaks = []
    if aes is None:
        aes = Aesthetics()
    if data is None:
        data = pd.DataFrame()

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)

    style = assign_identity_scales_get_style(
        aes=aes, p_color=color, p_size=size, p_marker=marker, p_alpha=alpha
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomPoint(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_error_bar(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "none_type",
    error_bar_kind: ERROR_BAR_KIND_VALUES = "linest",
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = -1,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
) -> "Geom":
    if aes is None:
        aes = Aesthetics()
    if data is None:
        data = pd.DataFrame()
    if breaks is None:
        breaks = []

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)
    error_bar_kind_ = ErrorBarKind.eitem(error_bar_kind)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=size,
        p_alpha=alpha,
        p_error_bar_kind=error_bar_kind_,
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomErrorBar(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_linerange(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "none_type",
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = -1,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
) -> "Geom":
    if aes is None:
        aes = Aesthetics()
    if data is None:
        data = pd.DataFrame()
    if breaks is None:
        breaks = []

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)

    style = assign_identity_scales_get_style(
        aes=aes, p_color=color, p_size=size, p_alpha=alpha
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomErrorBar(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_bar(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    alpha: Optional[float] = None,
    position: POSITION_VALUES = "stack",
    stat: STAT_TYPE_VALUES = "count",
) -> Geom:
    if aes is None:
        aes = Aesthetics()
    if data is None:
        data = pd.DataFrame()

    df_opt = data if len(data) > 0 else None

    position_ = PositionType.eitem(position)
    stat_ = StatType.eitem(stat)

    aes = deepcopy(aes)

    style = assign_identity_scales_get_style(
        aes,
        p_color=color,
        p_fill_color=color,
        p_alpha=alpha,
        p_line_type=LineType.SOLID,
        p_line_width=1.0,
    )

    gid = get_gid()

    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=BinPositionType.NONE,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomBar(gg_data=gg_data)
    return result


def geom_line(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "none_type",
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = -1,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
) -> "Geom":
    if breaks is None:
        breaks = []
    if data is None:
        data = pd.DataFrame()
    if aes is None:
        aes = Aesthetics()

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)
    line_type_ = LineType.eitem(line_type)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=size,
        p_alpha=alpha,
        p_fill_color=fill_color,
        p_line_type=line_type_,
        p_line_width=size,
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomLine(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_vline(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    xintercept: Optional[Union[Union[float, int], Iterable[Union[float, int]]]] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "none_type",
    fill_color: PossibleColor = None,
    position: POSITION_VALUES = "identity",
    alpha: Optional[float] = None,
    inhert_aes: bool = False,
) -> "Geom":

    if aes is None:
        aes = Aesthetics()

    if data is None:
        data = pd.DataFrame()

    if xintercept:
        stat_ = StatType.eitem("none")
    else:
        stat_ = StatType.eitem("identity")

    position_ = PositionType.eitem(position)
    line_type_ = LineType.eitem(line_type)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=size,
        p_alpha=alpha,
        p_fill_color=fill_color,
        p_line_type=line_type_,
        p_line_width=size,
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=data,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=None,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomVLine(gg_data=gg_data, xintercept=xintercept, inhert_aes=inhert_aes)

    return result


def geom_hline(
    yintercept: Optional[Union[Union[float, int], Iterable[Union[float, int]]]] = None,
    data: Optional[pd.DataFrame] = None,
    aes: Optional[Aesthetics] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "none_type",
    fill_color: PossibleColor = None,
    position: POSITION_VALUES = "identity",
    alpha: Optional[float] = None,
    inhert_aes: bool = False,
) -> "Geom":

    if aes is None:
        aes = Aesthetics()
    if data is None:
        data = pd.DataFrame()

    if yintercept:
        stat_ = StatType.eitem("none")
    else:
        stat_ = StatType.eitem("identity")

    position_ = PositionType.eitem(position)
    line_type_ = LineType.eitem(line_type)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=size,
        p_alpha=alpha,
        p_fill_color=fill_color,
        p_line_type=line_type_,
        p_line_width=size,
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=data,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=None,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomHLine(gg_data=gg_data, yintercept=yintercept, inhert_aes=inhert_aes)

    return result


def geom_abline(
    intercept: Union[int, float],
    slope: Union[int, float],
    aes: Optional[Aesthetics] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "none_type",
    fill_color: PossibleColor = None,
    position: POSITION_VALUES = "identity",
    alpha: Optional[float] = None,
    inhert_aes: bool = False,
) -> "Geom":

    if aes is None:
        aes = Aesthetics()

    stat_ = StatType.eitem("none")

    position_ = PositionType.eitem(position)
    line_type_ = LineType.eitem(line_type)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=size,
        p_alpha=alpha,
        p_fill_color=fill_color,
        p_line_type=line_type_,
        p_line_width=size,
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=None,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=None,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomABLine(
        gg_data=gg_data,
        intercept=intercept,
        slope=slope,
        inhert_aes=inhert_aes,
    )

    return result


def geom_rect(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    line_type: LINE_TYPE_VALUES = "solid",
    line_width: PossibleFloat = 0.5,
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "identity",
    position: POSITION_VALUES = "identity",
    alpha: Optional[float] = 1.0,
) -> "Geom":
    if data is None:
        data = pd.DataFrame()
    if aes is None:
        aes = Aesthetics()

    df_opt = data if len(data) > 0 else None
    stat_ = StatType.eitem(stat)
    position_ = PositionType.eitem(position)
    line_type_ = LineType.eitem(line_type)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=line_width,
        p_alpha=alpha,
        p_fill_color=fill_color,
        p_line_type=line_type_,
        p_line_width=line_width,
    )

    #  TODO in R ggplot x_min is xmin ETC
    required_attrs = [aes.x_min, aes.x_max, aes.y_min, aes.y_max]
    if None in required_attrs:
        raise GGException("Required all: 4 xmin, xmax, ymin, ymax")

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=None,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomRect(gg_data=gg_data)

    return result


def geom_area(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "none_type",
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "identity",
    bin_width: float = 0.0,
    bins: int = 30,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
) -> "Geom":
    if breaks is None:
        breaks = []
    if data is None:
        data = pd.DataFrame()
    if aes is None:
        aes = Aesthetics()

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)
    line_type_ = LineType.eitem(line_type)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=size,
        p_alpha=alpha,
        p_fill_color=fill_color,
        p_line_type=line_type_,
        p_line_width=size,
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomArea(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_smooth(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "none_type",
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = -1,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
    span: float = 0.7,
    smoother: SMOOTH_METHOD_TYPE_VALUES = "svg",
    poly_ordder: int = 5,
) -> "Geom":
    if breaks is None:
        breaks = []
    if data is None:
        data = pd.DataFrame()
    if aes is None:
        aes = Aesthetics()

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)
    line_type_ = LineType.eitem(line_type)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=size,
        p_alpha=alpha,
        p_line_type=line_type_,
        p_line_width=size,
    )
    stat_kind = StatSmooth(
        span=span, poly_order=poly_ordder, method_type=SmoothMethodType.eitem(smoother)
    )
    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=stat_kind,
        position=position_,
    )
    result = GeomLine(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_histogram(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    line_type: LINE_TYPE_VALUES = "solid",
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "bin",
    bins: int = 30,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "left",
    position: POSITION_VALUES = "stack",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
    line_width: PossibleFloat = 0.2,
    drawing_style: HISTOGRAM_DRAWING_STYLE_VALUES = "bars",
) -> "Geom":
    if aes is None:
        aes = Aesthetics()
    if data is None:
        data = pd.DataFrame()
    if breaks is None:
        breaks = []
    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)
    line_type_ = LineType.eitem(line_type)
    drawing_style_ = HistogramDrawingStyle.eitem(drawing_style)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_fill_color=fill_color,
        p_line_width=size,
        p_line_type=line_type_,
        p_alpha=alpha,
    )
    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomHistogram(gg_data=gg_data, histogram_drawing_style=drawing_style_)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_freqpoly(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    alpha: Optional[float] = None,
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "bin",
    bins: int = 30,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "center",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    line_type: LINE_TYPE_VALUES = "solid",
) -> "Geom":
    if breaks is None:
        breaks = []
    if data is None:
        data = pd.DataFrame()
    if aes is None:
        aes = Aesthetics()

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)
    line_type_ = LineType.eitem(line_type)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_fill_color=fill_color,
        p_line_width=size,
        p_line_type=line_type_,
        p_alpha=alpha,
    )
    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomFreqPoly(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_tile(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = 30,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
    line_width: PossibleFloat = 0.2,
) -> "Geom":
    if breaks is None:
        breaks = []
    if data is None:
        data = pd.DataFrame()
    if aes is None:
        aes = Aesthetics()

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_fill_color=fill_color,
        p_line_width=size,
        p_alpha=alpha,
    )
    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomTile(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_raster(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = 30,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
    line_width: PossibleFloat = 0.2,
) -> "Geom":
    if breaks is None:
        breaks = []
    if data is None:
        data = pd.DataFrame()
    if aes is None:
        aes = Aesthetics()

    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_fill_color=fill_color,
        p_line_width=size,
        p_alpha=alpha,
    )
    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomRaster(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result


def geom_text(
    aes: Optional[Aesthetics] = None,
    data: Optional[pd.DataFrame] = None,
    color: PossibleColor = None,
    size: PossibleFloat = None,
    fill_color: PossibleColor = None,
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = -1,
    bin_width: float = 0.0,
    breaks: Optional[List[float]] = None,
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
    line_width: PossibleFloat = 0.2,
) -> "Geom":
    if breaks is None:
        breaks = []
    if aes is None:
        aes = Aesthetics()
    if data is None:
        data = pd.DataFrame()
    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)

    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_fill_color=fill_color,
        p_line_width=size,
        p_alpha=alpha,
    )
    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),
        position=position_,
    )
    result = GeomText(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result
