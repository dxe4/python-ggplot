from copy import deepcopy
from typing import Any

from python_ggplot.core.objects import AxisKind
from python_ggplot.gg.geom.base import Geom
from python_ggplot.gg.scales.base import DateScale, GGScale
from python_ggplot.gg.types import Aesthetics, Annotation, Facet, GgPlot, Ridges, Theme
from python_ggplot.public_interface.utils import apply_scale, apply_theme


def add_scale(plot: GgPlot, scale: GGScale) -> GgPlot:
    # TODO MEDIUM priority EASY task
    # does this need deep copy?
    result = deepcopy(plot)

    result.aes = apply_scale(result.aes, scale)
    for geom in result.geoms:
        geom.gg_data.aes = apply_scale(geom.gg_data.aes, scale)
    return result


def add_date_scale(p: GgPlot, date_scale: DateScale) -> GgPlot:
    """
    TODO refactor....
    this is a bit of a mess but *should* work
    """

    def assign_copy_scale(obj: Aesthetics, field: str, ds: Any):
        field_val = obj.__dict__.get(field)
        if field_val is not None:
            scale = deepcopy(field_val)
            scale.date_scale = ds
            setattr(obj, field, scale)

    result = deepcopy(p)

    if date_scale.axis_kind == AxisKind.X:
        assign_copy_scale(result.aes, "x", date_scale)
    elif date_scale.axis_kind == AxisKind.Y:
        assign_copy_scale(result.aes, "y", date_scale)

    for geom in result.geoms:
        if date_scale.axis_kind == AxisKind.X:
            assign_copy_scale(geom.gg_data.aes, "x", date_scale)
        elif date_scale.axis_kind == AxisKind.Y:
            assign_copy_scale(geom.gg_data.aes, "y", date_scale)

    return result


def add_theme(plot: GgPlot, theme: Theme) -> GgPlot:
    apply_theme(plot.theme, theme)

    if plot.theme.title is not None:
        plot.title = plot.theme.title
    if plot.theme.sub_title is not None:
        plot.title = plot.theme.sub_title

    return plot


def add_geom(plot: GgPlot, geom: Geom) -> GgPlot:
    plot.geoms.append(geom)
    return plot


def add_facet(plot: GgPlot, facet: Facet) -> GgPlot:
    plot.facet = facet
    return plot


def add_ridges(plot: GgPlot, ridges: Ridges) -> GgPlot:
    plot.ridges = ridges
    return plot


def add_annotations(plot: GgPlot, annotations: Annotation) -> GgPlot:
    plot.annotations.append(annotations)
    return plot
