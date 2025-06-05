from typing import Any, Dict, List

from python_ggplot.gg.scales.base import FilledScales, GGScale, MainAddScales, ScaleType
from python_ggplot.gg.scales.collect_and_fill import (
    FillScaleData,
    add_facets,
    call_fill_scale,
)
from python_ggplot.gg.types import GgPlot


def collect(plot: GgPlot, field_name: str) -> List[FillScaleData]:
    scale_data: List[FillScaleData] = []

    attr_value = getattr(plot.aes, field_name, None)
    if attr_value is not None:
        element = FillScaleData(df=None, scale=attr_value)
        scale_data.append(element)

    for geom in plot.geoms:
        geom_aes = getattr(geom.gg_data.aes, field_name, None)
        if geom_aes is None:
            continue

        element = FillScaleData(df=geom.gg_data.data, scale=geom_aes)
        scale_data.append(element)

    return scale_data


def collect_scales(plot: GgPlot) -> FilledScales:
    result: Dict[Any, Any] = {}

    def fill_field(field_name: str, arg: List[GGScale]) -> None:
        if len(arg) > 0 and arg[0].gg_data.ids == set(range(0, 65536)):  # type: ignore
            result[field_name] = MainAddScales(main=arg[0], more=arg[1:])
        else:
            result[field_name] = MainAddScales(main=None, more=arg)

    xs = collect(plot, "x")
    x_filled = call_fill_scale(plot.data, xs, ScaleType.LINEAR_DATA)
    fill_field("x", x_filled)

    if any(x.scale.is_reversed() for x in xs):
        result["reversed_x"] = True
    if any(x.is_discrete() for x in x_filled):
        result["discrete_x"] = True

    xintercept = collect(plot, "xintercept")
    xintercept_filled = call_fill_scale(plot.data, xintercept, ScaleType.LINEAR_DATA)
    fill_field("xintercept", xintercept_filled)

    yintercept = collect(plot, "yintercept")
    yintercept_filled = call_fill_scale(plot.data, yintercept, ScaleType.LINEAR_DATA)
    fill_field("yintercept", yintercept_filled)

    xs_min = collect(plot, "x_min")
    x_min_filled = call_fill_scale(plot.data, xs_min, ScaleType.LINEAR_DATA)
    fill_field("x_min", x_min_filled)

    xs_max = collect(plot, "x_max")
    x_max_filled = call_fill_scale(plot.data, xs_max, ScaleType.LINEAR_DATA)
    fill_field("x_max", x_max_filled)

    ys = collect(plot, "y")
    y_filled = call_fill_scale(plot.data, ys, ScaleType.LINEAR_DATA)
    fill_field("y", y_filled)

    if any(y.scale.is_reversed() for y in ys):
        result["reversed_y"] = True
    if any(y.is_discrete() for y in y_filled):
        result["discrete_y"] = True

    ys_min = collect(plot, "y_min")
    y_min_filled = call_fill_scale(plot.data, ys_min, ScaleType.LINEAR_DATA)
    fill_field("y_min", y_min_filled)

    # Handle y_max scales
    ys_max = collect(plot, "y_max")
    y_max_filled = call_fill_scale(plot.data, ys_max, ScaleType.LINEAR_DATA)
    fill_field("y_max", y_max_filled)

    # Handle y_ridges scales
    ys_ridges = collect(plot, "y_ridges")
    y_ridges_filled = call_fill_scale(plot.data, ys_ridges, ScaleType.LINEAR_DATA)
    fill_field("y_ridges", y_ridges_filled)

    # Handle color scales
    colors = collect(plot, "color")
    color_filled = call_fill_scale(plot.data, colors, ScaleType.COLOR)
    fill_field("color", color_filled)

    # Handle fill scales
    fills = collect(plot, "fill")
    fill_filled = call_fill_scale(plot.data, fills, ScaleType.FILL_COLOR)
    fill_field("fill", fill_filled)

    # Handle alpha scales
    alphas = collect(plot, "alpha")
    alpha_filled = call_fill_scale(plot.data, alphas, ScaleType.ALPHA)
    fill_field("alpha", alpha_filled)

    # Handle size scales
    sizes = collect(plot, "size")
    size_filled = call_fill_scale(plot.data, sizes, ScaleType.SIZE)
    fill_field("size", size_filled)

    # Handle shape scales
    shapes = collect(plot, "shape")
    shape_filled = call_fill_scale(plot.data, shapes, ScaleType.SHAPE)
    fill_field("shape", shape_filled)

    # Handle width scales
    widths = collect(plot, "width")
    width_filled = call_fill_scale(plot.data, widths, ScaleType.LINEAR_DATA)
    fill_field("width", width_filled)

    # Handle height scales
    heights = collect(plot, "height")
    height_filled = call_fill_scale(plot.data, heights, ScaleType.LINEAR_DATA)
    fill_field("height", height_filled)

    # Handle text scales (dummy scale, only care about column)
    texts = collect(plot, "text")
    text_filled = call_fill_scale(plot.data, texts, ScaleType.TEXT)
    fill_field("text", text_filled)

    # Handle weight scales
    weights = collect(plot, "weight")
    weight_filled = call_fill_scale(plot.data, weights, ScaleType.LINEAR_DATA)
    fill_field("weight", weight_filled)

    filled_scales_result = FilledScales(**result)  # type: ignore
    if plot.facet is not None:
        add_facets(filled_scales_result, plot)

    return filled_scales_result
