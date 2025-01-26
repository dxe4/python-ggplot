from typing import Set

from python_ggplot.core.common import GREY20, TRANSPARENT, WHITE
from python_ggplot.core.objects import AxisKind, Color, LineType, Scale, Style
from python_ggplot.gg_geom import FilledScales
from python_ggplot.gg_scales import get_secondary_axis, get_x_scale, get_y_scale
from python_ggplot.gg_types import GgPlot, StatBin, Theme


def get_plot_background(theme: Theme) -> Style:
    result = Style(color=Color(**TRANSPARENT))
    result.fill_color = theme.plot_background_color or Color(**GREY20)
    return result


def get_canvas_background(theme: Theme) -> Style:
    result = Style(color=Color(**TRANSPARENT))
    result.fill_color = theme.canvas_color or Color(**WHITE)
    return result


def get_grid_line_style(theme: Theme) -> Style:
    result = Style(
        line_width=theme.grid_line_width or 1.0,
        color=theme.grid_line_color or Color(**WHITE),
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
    x_scale = get_x_scale(filled_scales)
    if x_scale.name:
        return x_scale.name

    # TODO high priority sanity check what str should return
    return str(x_scale.col)


def _get_y_label(filled_scales: "FilledScales") -> str:
    y_scale = get_y_scale(filled_scales)
    if y_scale.name:
        return y_scale.name
    elif y_scale.col.name:
        return str(y_scale.col)
    else:
        stat_types: Set[StatBin] = {
            filled_scales.geom.stat_kind
            for filled_scales in filled_scales.geoms
            if isinstance(filled_scales.geom.stat_kind, StatBin)
            and filled_scales.geom.stat_kind.density
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
        theme.x_label_secondary = get_secondary_axis(filled_scales, AxisKind.X).name
    if theme.y_label_secondary is None and has_secondary(theme, AxisKind.Y):
        theme.y_label_secondary = get_secondary_axis(filled_scales, AxisKind.Y).name

    x_scale = theme.x_range or filled_scales.x_scale
    theme.x_margin_range = calculate_margin_range(theme, x_scale, AxisKind.X)

    y_scale = theme.y_range or filled_scales.y_scale
    theme.y_margin_range = calculate_margin_range(theme, y_scale, AxisKind.Y)

    return theme
