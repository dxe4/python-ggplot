from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, cast

from python_ggplot.core.objects import GGException
from python_ggplot.core.units.objects import AxisKind, UnitType
from python_ggplot.graphics.objects import (
    GOAxis,
    GOGrid,
    GOLabel,
    GOLine,
    GOManyPoints,
    GOPoint,
    GOPolyLine,
    GORaster,
    GORect,
    GOText,
    GOTick,
    GOTickLabel,
    GOType,
    GraphicsObject,
)

if TYPE_CHECKING:
    from python_ggplot.graphics.views import ViewPort


@dataclass
class GOConvertData:
    graphics_obj: GraphicsObject
    view: Optional["ViewPort"]
    axis: Optional[AxisKind]


def star_stop_to_relative(data: GOConvertData) -> GraphicsObject:
    if not isinstance(data.graphics_obj, (GOLine, GOAxis)):
        raise GGException("unexpected")

    obj = data.graphics_obj
    obj.data.start = obj.data.start.to_relative()
    obj.data.stop = obj.data.stop.to_relative()
    return obj


def text_to_relative(data: GOConvertData) -> GraphicsObject:
    if not isinstance(data.graphics_obj, (GOText, GOLabel, GOTickLabel)):
        raise GGException("unexpected")

    obj = data.graphics_obj
    obj.data.pos = obj.data.pos.to_relative()
    return obj


def rect_to_relative(data: GOConvertData) -> GraphicsObject:
    obj = cast(GORect, data.graphics_obj)
    obj.origin = obj.origin.to_relative()
    obj.width = obj.width.to_relative()
    obj.height = obj.height.to_relative()
    return obj


def raster_to_relative(data: GOConvertData) -> GraphicsObject:
    if data.axis is None or data.view is None:
        raise GGException("expected view and axis")

    obj = cast(GORaster, data.graphics_obj)
    obj.origin = obj.origin.to_relative()
    obj.pixel_width = obj.pixel_width.to_relative_from_view(data.view, data.axis)
    obj.pixel_height = obj.pixel_height.to_relative_from_view(data.view, data.axis)
    return obj


def grid_to_relative(data: GOConvertData) -> GraphicsObject:
    if data.view is None:
        raise GGException("expected view")

    obj = cast(GOGrid, data.graphics_obj)
    obj.origin = obj.origin.to_relative()
    obj.origin_diagonal = obj.origin_diagonal.to_relative()

    height = data.view.point_height()
    width = data.view.point_width()
    obj.x_pos = [i.to_relative(width) for i in obj.x_pos]
    obj.y_pos = [i.to_relative(height) for i in obj.y_pos]

    return obj


def tick_to_relative(data: GOConvertData) -> GraphicsObject:
    obj = cast(GOTick, data.graphics_obj)
    obj.pos = obj.pos.to_relative()
    return obj


def point_to_relative(data: GOConvertData) -> GraphicsObject:
    obj = cast(GOPoint, data.graphics_obj)
    obj.pos = obj.pos.to_relative()
    return obj


def many_points_to_relative(data: GOConvertData) -> GraphicsObject:
    obj = cast(GOManyPoints, data.graphics_obj)
    obj.pos = [i.to_relative() for i in obj.pos]
    return obj


def polyline_to_relative(data: GOConvertData) -> GraphicsObject:
    obj = cast(GOPolyLine, data.graphics_obj)
    obj.pos = [i.to_relative() for i in obj.pos]
    return obj


lookup_map = {
    # start/stop
    (GOType.AXIS, UnitType.RELATIVE): star_stop_to_relative,
    (GOType.LINE, UnitType.RELATIVE): star_stop_to_relative,
    # text
    (GOType.TEXT, UnitType.RELATIVE): text_to_relative,
    (GOType.LABEL, UnitType.RELATIVE): text_to_relative,
    (GOType.TICK_LABEL, UnitType.RELATIVE): text_to_relative,
    # others
    (GOType.GRID_DATA, UnitType.RELATIVE): grid_to_relative,
    (GOType.TICK_DATA, UnitType.RELATIVE): tick_to_relative,
    (GOType.POINT_DATA, UnitType.RELATIVE): point_to_relative,
    (GOType.MANY_POINTS_DATA, UnitType.RELATIVE): many_points_to_relative,
    (GOType.POLYLINE_DATA, UnitType.RELATIVE): polyline_to_relative,
    (GOType.RECT_DATA, UnitType.RELATIVE): rect_to_relative,
    (GOType.RASTER_DATA, UnitType.RELATIVE): raster_to_relative,
    # (GOType.COMPOSITE_DATA, UnitType.RELATIVE): None,
}


def graphics_object_to_relative(
    graphics_obj: GraphicsObject,
    view: Optional["ViewPort"] = None,
    axis: Optional[AxisKind] = None,
) -> GraphicsObject:
    func = lookup_map.get((graphics_obj.go_type, UnitType.RELATIVE))
    if not func:
        raise GGException("Conversion not possible")

    data = GOConvertData(graphics_obj=graphics_obj, view=view, axis=axis)
    return func(data)
