from typing import Set

from python_ggplot.core.chroma import to_opt_color
from python_ggplot.core.objects import (
    GREY92_DICT,
    TRANSPARENT_DICT,
    WHITE_DICT,
    AxisKind,
    Color,
    GGException,
    LineType,
    Scale,
    Style,
)
from python_ggplot.gg.scales.base import FilledScales
from python_ggplot.gg.types import GgPlot, StatBin, Theme


def get_plot_background(theme: Theme) -> Style:
    result = Style(color=Color(**TRANSPARENT_DICT))
    result.fill_color = to_opt_color(theme.plot_background_color) or Color(
        **GREY92_DICT
    )
    return result


def get_canvas_background(theme: Theme) -> Style:
    result = Style(color=Color(**TRANSPARENT_DICT))
    result.fill_color = to_opt_color(theme.canvas_color) or Color(**WHITE_DICT)
    return result


def get_grid_line_style(theme: Theme) -> Style:
    result = Style(
        line_width=theme.grid_line_width or 1.0,
        color=to_opt_color(theme.grid_line_color) or Color(**WHITE_DICT),
        line_type=LineType.SOLID,
    )
    return result


def get_minor_grid_line_style(major_style: Style, theme: Theme) -> Style:
    result = Style(
        line_width=theme.minor_grid_line_width or major_style.line_width / 2.0,
        color=major_style.color,
        line_type=LineType.SOLID,
    )
    return result


def calculate_margin_range(theme: Theme, scale: Scale, ax_kind: AxisKind) -> Scale:
    margin_lookup = {
        AxisKind.X: theme.x_margin or 0.0,
        AxisKind.Y: theme.y_margin or 0.0,
    }
    margin = margin_lookup[ax_kind]
    diff = scale.high - scale.low
    return Scale(low=scale.low - diff * margin, high=scale.high + diff * margin)


def has_secondary(theme: "Theme", ax_kind: AxisKind) -> bool:
    secondary_lookup = {
        AxisKind.X: lambda: theme.x_label_secondary is not None,
        AxisKind.Y: lambda: theme.y_label_secondary is not None,
    }
    return secondary_lookup[ax_kind]()


def label_name(filled_scales: "FilledScales", p: "GgPlot", ax_kind: AxisKind) -> str:
    label_lookup = {
        AxisKind.X: lambda: _get_x_label(filled_scales),
        AxisKind.Y: lambda: _get_y_label(filled_scales),
    }
    return label_lookup[ax_kind]()


def _get_x_label(filled_scales: "FilledScales") -> str:
    return filled_scales.x.get_name()


def _get_y_label(filled_scales: "FilledScales") -> str:
    """
    TODO CRITICAL, medium difficulty
    """
    y_scale = filled_scales.y
    if y_scale is not None and (y_scale.main or y_scale.more):
        return filled_scales.y.get_name()
    else:
        stat_types: Set[StatBin] = {
            filled_scales.gg_data.geom.gg_data.stat_kind
            for filled_scales in filled_scales.geoms
            if isinstance(filled_scales.gg_data.geom.gg_data.stat_kind, StatBin)
            and filled_scales.gg_data.geom.gg_data.stat_kind.density
        }  # type: ignore TODO

        if stat_types:
            return "density"
        else:
            return "count"


def build_theme(filled_scales: FilledScales, plot: GgPlot) -> "Theme":
    theme = plot.theme

    if theme.x_label is None:
        theme.x_label = label_name(filled_scales, plot, AxisKind.X)
    if theme.y_label is None:
        theme.y_label = label_name(filled_scales, plot, AxisKind.Y)

    if theme.x_label_secondary is None and has_secondary(theme, AxisKind.X):
        theme.x_label_secondary = filled_scales.get_secondary_axis(AxisKind.X).name
    if theme.y_label_secondary is None and has_secondary(theme, AxisKind.Y):
        theme.y_label_secondary = filled_scales.get_secondary_axis(AxisKind.Y).name

    x_scale = theme.x_range or filled_scales.x_scale
    y_scale = theme.y_range or filled_scales.y_scale

    theme.x_margin_range = calculate_margin_range(theme, x_scale, AxisKind.X)
    theme.y_margin_range = calculate_margin_range(theme, y_scale, AxisKind.Y)

    return theme
