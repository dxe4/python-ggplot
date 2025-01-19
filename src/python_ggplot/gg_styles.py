from python_ggplot.core.chroma import parse_hex
from python_ggplot.core.objects import (
    BLACK,
    GREY20,
    TRANSPARENT,
    Color,
    Font,
    LineType,
    MarkerKind,
    Style,
)

# Define color constants
STAT_SMOOTH_COLOR = Color(
    **parse_hex("#3366FF")
)  # color used by ggplot2 for smoothed lines


# Define default styles
POINT_DEFAULT_STYLE = Style(
    size=3.0, marker=MarkerKind.CIRCLE, color=BLACK, fill_color=BLACK
)

LINE_DEFAULT_STYLE = Style(
    line_width=1.0,
    line_type=LineType.SOLID,
    size=5.0,
    color=GREY20,
    fill_color=TRANSPARENT,
)

SMOOTH_DEFAULT_STYLE = Style(
    line_width=2.0,
    line_type=LineType.SOLID,
    size=5.0,
    color=STAT_SMOOTH_COLOR,
    fill_color=TRANSPARENT,
)

BAR_DEFAULT_STYLE = Style(
    line_width=1.0, line_type=LineType.SOLID, color=GREY20, fill_color=GREY20
)

HISTO_DEFAULT_STYLE = Style(
    line_width=0.2, line_type=LineType.SOLID, color=GREY20, fill_color=GREY20
)

TILE_DEFAULT_STYLE = Style(
    line_width=0.05, line_type=LineType.SOLID, color=GREY20, fill_color=GREY20
)

TEXT_DEFAULT_STYLE = Style(
    # TODO, there is a macro for font in ggplot, we may have to pick the defualt values from there
    font=Font(size=12.0),
    size=12.0,
    color=BLACK,
)

# Define ranges
DEFAULT_SIZE_RANGE = {"low": 2.0, "high": 7.0}
DEFAULT_ALPHA_RANGE = {"low": 0.1, "high": 1.0}

# todo this needs the color maps to be ported
# DefaultColorScale* = viridis()
