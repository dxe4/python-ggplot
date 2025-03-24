from math import ceil, sqrt
from types import NoneType
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from python_ggplot.core.chroma import to_opt_color
from python_ggplot.core.objects import (
    AxisKind,
    Color,
    ErrorBarKind,
    Font,
    GGException,
    LineType,
    MarkerKind,
)
from python_ggplot.gg.types import (
    Aesthetics,
    GGStyle,
    PossibleColor,
    PossibleErrorBar,
    PossibleFloat,
    PossibleFont,
    PossibleLineType,
    PossibleMarker,
    SecondaryAxis,
)


def calc_rows_columns(rows: int, columns: int, n_plots: int) -> Tuple[int, int]:
    if rows <= 0 and columns <= 0:
        sq_plt = sqrt(float(n_plots))
        return (round(sq_plt), ceil(sq_plt))
    elif rows == -1 and columns > 0:
        return (ceil(n_plots / columns), columns)
    elif rows > 0 and columns == -1:
        return (rows, ceil(n_plots / rows))
    elif rows == 0 and columns > 0:
        return (1, columns)
    elif rows > 0 and columns == 0:
        return (rows, 1)
    else:
        return (rows, columns)


def _handle_none(_: Any) -> None:
    return None


def _handle_float(x: Any) -> float:
    return float(x)


def _handle_bool(x: Any) -> bool:
    return bool(x)


def _handle_gg_type(x: Any) -> Any:
    return x


def _handle_optional(x: Optional[Any]) -> Any:
    return x


FLOAT_HANDLERS: Dict[type, Callable[[Any], Any]] = {
    int: _handle_float,
    float: _handle_float,
    NoneType: _handle_optional,
}

BOOL_HANDLERS: Dict[type, Callable[[Any], Any]] = {
    NoneType: _handle_none,
    bool: _handle_bool,
}

GG_TYPE_HANDLERS: Dict[type, Callable[[Any], Any]] = {
    ErrorBarKind: _handle_gg_type,
    MarkerKind: _handle_gg_type,
    LineType: _handle_gg_type,
    NoneType: _handle_optional,
}


def to_opt_float(x: Any) -> Optional[float]:
    return FLOAT_HANDLERS.get(type(x))(x)  # type: ignore


def to_opt_bool(x: Any) -> Optional[bool]:
    return BOOL_HANDLERS.get(type(x))(x)  # type: ignore


def to_opt_marker(x: Any) -> Optional[MarkerKind]:
    return GG_TYPE_HANDLERS.get(type(x))(x)  # type: ignore


def to_opt_line_type(x: Any) -> Optional[LineType]:
    return GG_TYPE_HANDLERS.get(type(x))(x)  # type: ignore


def to_opt_error_bar(x: Any) -> Optional[ErrorBarKind]:
    return GG_TYPE_HANDLERS.get(type(x))(x)  # type: ignore


def to_opt_font(x: Any) -> Optional[Font]:
    return GG_TYPE_HANDLERS.get(type(x))(x)  # type: ignore


def to_opt_sec_axis(
    x: Optional[SecondaryAxis], axis: AxisKind
) -> Optional[SecondaryAxis]:
    if x is None:
        return None
    x.axis_kind = axis

    return x


def _handle_style(
    aes: Aesthetics,
    x: Any,
    to_opt_func: Callable[[Any], Any],
    scale_func: Callable[[Any], Any],
    attr_name: str,
) -> Optional[Any]:
    result = to_opt_func(x)
    if result is None and isinstance(x, str):
        if len(x) == 0:
            raise GGException("Don't hand an empty string as a column reference!")
        setattr(aes, attr_name, scale_func(x))
    return result


def scale_color_identity(x: Any):
    # TODO high priority implement
    raise GGException("")


# TDODO refactoron those or functions
def color_or_style(aes: Aesthetics, x: Any) -> Optional[Any]:
    color = to_opt_color(x)
    if color:
        return color

    if isinstance(x, str):
        aes.color = scale_color_identity(x)


def fill_or_style(aes: Aesthetics, x: Any) -> Optional[Any]:
    color = to_opt_color(x)
    if color:
        return color

    if isinstance(x, str):
        aes.fill = scale_color_identity(x)


def size_or_style(aes: Aesthetics, x: Any) -> Optional[float]:
    color = to_opt_float(x)
    if color:
        return color

    if isinstance(x, str):
        aes.size = scale_color_identity(x)


def alpha_or_style(aes: Aesthetics, x: Any) -> Optional[float]:
    color = to_opt_float(x)
    if color:
        return color

    if isinstance(x, str):
        aes.alpha = scale_color_identity(x)


def init_gg_style(
    color: Optional[Color] = None,
    size: Optional[float] = None,
    marker: Optional[MarkerKind] = None,
    line_type: Optional[LineType] = None,
    line_width: Optional[float] = None,
    fill_color: Optional[Color] = None,
    error_bar_kind: Optional[ErrorBarKind] = None,
    alpha: Optional[float] = None,
    font: Optional[Font] = None,
) -> GGStyle:
    # TODO redundant, keep for compatibility for now
    return GGStyle(
        color=color,
        size=size,
        marker=marker,
        line_type=line_type,
        line_width=line_width,
        fill_color=fill_color,
        error_bar_kind=error_bar_kind,
        alpha=alpha,
        font=font,
    )


def assign_identity_scales_get_style(
    aes: Aesthetics,
    p_color: Optional[PossibleColor] = None,
    p_size: Optional[PossibleFloat] = None,
    p_marker: Optional[PossibleMarker] = None,
    p_line_type: Optional[PossibleLineType] = None,
    p_line_width: Optional[PossibleFloat] = None,
    p_fill_color: Optional[PossibleColor] = None,
    p_error_bar_kind: Optional[PossibleErrorBar] = None,
    p_alpha: Optional[PossibleFloat] = None,
    p_font: Optional[PossibleFont] = None,
) -> GGStyle:

    color = color_or_style(aes, p_color)
    fill = fill_or_style(aes, p_fill_color)
    size = size_or_style(aes, p_size)
    alpha = alpha_or_style(aes, p_alpha)

    marker = to_opt_marker(p_marker)
    line_type = to_opt_line_type(p_line_type)
    line_width = to_opt_float(p_line_width)
    error_bar = to_opt_error_bar(p_error_bar_kind)
    font = to_opt_font(p_font)
    return init_gg_style(
        color, size, marker, line_type, line_width, fill, error_bar, alpha, font
    )
