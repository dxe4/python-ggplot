"""
one solution would be to write some code that auto generates this file
get all subclasses of GGenum, loop through them and generate this.
sounds like an easy job
i guess this involves AST?

this approach is not great for a developer approach
but its very convinient for the users and for documentation
i cant imagine anyone being happy doing:
    geom_point(position=PositionType.IDENTITY)
    isntead of geom_point(position="identidy)
especially over multiple variables
"""

from typing import Literal

STAT_TYPE_VALUES = Literal["identity", "count", "bin", "smooth"]
BIN_POSITION_VALUES = Literal["none", "center", "left", "right"]
POSITION_VALUES = Literal["identity", "stack", "dodge", "fill"]
BIN_BY_VALUES = Literal["full", "subset"]
OPRATOR_TYPE_VALUES = Literal["div", "add", "sub", "mul"]
MARKER_KIND_VALUES = Literal[
    "circle",
    "cross",
    "triangle",
    "rhombus",
    "rectangle",
    "rotcross",
    "upsidedown_triangle",
    "empty_circle",
    "empty_rectangle",
    "empty_rhombus",
]
FILE_TYPE_KIND_VALUES = Literal[
    "svg",
    "png",
    "pdf",
    "vega",
    "tex",
]
LINE_TYPE_VALUES = Literal[
    "none_type",
    "solid",
    "dashed",
    "dotted",
    "dot_dash",
    "long_dash",
    "two_dash",
]

ERROR_BAR_KIND_VALUES = Literal[
    "lines",
    "linest",
]

TEXT_ALIGN_KIND_VALUES = Literal[
    "left",
    "center",
    "right",
]

CFONT_SLANT_VALUES = Literal[
    "normal",
    "italic",
    "oblique",
]

AXIS_KIND_VALUES = Literal["X", "Y"]

COMPOSITE_KIND_VALUES = Literal["error_bar",]

TICK_KIND_VALUES = Literal[
    "one_side",
    "both_sides",
]

UNIT_TYPE_VALUES = Literal[
    "point",
    "centimeter",
    "inch",
    "relative",
    "data",
    "str_width",
    "str_height",
    "abstract",
]

COLUMN_TYPE_VALUES = Literal[
    "none",
    "float",
    "int",
    "bool",
    "string",
    "object",
    "constant",
    "generic",
]

FORMULA_TYPE_VALUES = Literal[
    "variable",
    "assign",
    "vector",
    "scalar",
    "none",
]

GEOM_TYPE_VALUES = Literal[
    "point",
    "bar",
    "histogram",
    "freq_poly",
    "tile",
    "line",
    "error_bar",
    "text",
    "raster",
]

HISTOGRAM_DRAWING_STYLE_VALUES = Literal[
    "bars",
    "outline",
]


SCALE_TYPE_VALUES = Literal[
    "linear_data",
    "transformed_data",
    "color",
    "fill_color",
    "alpha",
    "shape",
    "size",
    "text",
]

SCALE_FREE_KIND_VALUES = Literal[
    "fixed",
    "free_x",
    "free_y",
    "free",
]

POSITION_TYPE_VALUES = Literal[
    "identity",
    "stack",
    "dodge",
    "fill",
]

DISCRETE_TYPE_VALUES = Literal[
    "discrete",
    "continuous",
]

DATE_TICK_ALGORITHM_TYPE_VALUES = Literal[
    "filter",
    "add_duration",
    "custom_breaks",
]

DATA_TYPE_VALUES = Literal[
    "mapping",
    "setting",
    "null",
]

BIN_POSITION_TYPE_VALUES = Literal[
    "none",
    "center",
    "left",
    "right",
]

SMOOTH_METHOD_TYPE_VALUES = Literal[
    "svg",
    "lm",
    "poly",
]

BIN_BY_TYPE_VALUES = Literal[
    "full",
    "subset",
]

OUTSIDE_RANGE_KIND_VALUES = Literal[
    "none",
    "drop",
    "clip",
]

VEGA_BACKEND_VALUES = Literal[
    "webview",
    "browser",
]

GO_TYPE_VALUES = Literal[
    "line",
    "axis",
    "text",
    "tick_label",
    "label",
    "grid_data",
    "tick_data",
    "point_data",
    "many_points_data",
    "polyline_data",
    "rect_data",
    "raster_data",
    "composite_data",
]
