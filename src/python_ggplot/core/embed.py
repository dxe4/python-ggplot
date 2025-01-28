from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from python_ggplot.core.coord.objects import Coord, Coord1D, RelativeCoordType
from python_ggplot.core.objects import AxisKind, GGException, UnitType
from python_ggplot.core.units.objects import Quantity
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


# coord
def _coord_embed_into_origin_for_length(view: "ViewPort", axis_kind: AxisKind):
    data = {
        AxisKind.X: (view.origin.x, view.w_img),
        AxisKind.Y: (view.origin.y, view.h_img),
    }
    return data[axis_kind]


def _coord_embed_into_origin(view: "ViewPort", axis_kind: AxisKind):
    data = {
        AxisKind.X: (view.left(), view.get_width()),
        AxisKind.Y: (view.bottom(), view.get_height()),
    }
    return data[axis_kind]


def coord1d_embed_into(
    coord: Coord1D, axis_kind: AxisKind, into: "ViewPort"
) -> Coord1D:
    if coord.unit_type.is_length():
        origin, abs_length = _coord_embed_into_origin_for_length(into, axis_kind)
        origin_abs = origin.to_via_points(UnitType.POINT, abs_length=abs_length)
        return origin_abs + coord
    else:
        origin, abs_length = _coord_embed_into_origin(into, axis_kind)
        pos = origin.pos + abs_length.val * coord.to_relative().pos
        return RelativeCoordType(pos)


def coord_embed_into(coord: Coord, into: "ViewPort") -> Coord:
    return Coord(
        x=coord.x.embed_into(AxisKind.X, into),
        y=coord.y.embed_into(AxisKind.Y, into),
    )


# Quantity
def quantity_embed_into_origin(view: "ViewPort", axis_kind: AxisKind):
    if axis_kind == AxisKind.X:
        return view.width, view.point_width(), view.x_scale
    if axis_kind == AxisKind.Y:
        return view.height, view.point_height(), view.y_scale
    raise GGException("unexpected")


def relative_quantity_embed_into(
    quantity: Quantity, axis: AxisKind, view: "ViewPort"
) -> "Quantity":
    new_quantity, length, scale = quantity_embed_into_origin(view, axis)
    result = quantity.multiply(
        new_quantity, length=length, scale=scale, as_coordinate=False
    )
    return result


def data_quantity_embed_into(
    quantity: Quantity, axis: AxisKind, view: "ViewPort"
) -> "Quantity":
    new_quantity, length, scale = quantity_embed_into_origin(view, axis)
    result = quantity.multiply(
        new_quantity, length=length, scale=scale, as_coordinate=False
    )
    return result


quantity_embed_into_lookup = {
    UnitType.RELATIVE: relative_quantity_embed_into,
    UnitType.DATA: data_quantity_embed_into,
}


def quantity_embed_into(
    quantity: Quantity, axis: AxisKind, view: "ViewPort"
) -> Quantity:
    if quantity.unit_type.is_length():
        return deepcopy(quantity)
    func = quantity_embed_into_lookup.get(quantity.unit_type)

    if not func:
        raise GGException("cannot embed quantity")

    return func(quantity, axis, view)


# view
def view_embed_into(current_view: "ViewPort", into: "ViewPort") -> "ViewPort":
    current_view.origin = current_view.origin.embed_into(into)
    current_view.height = current_view.height.embed_into(AxisKind.Y, into)
    current_view.width = current_view.width.embed_into(AxisKind.X, into)
    return current_view


def view_embed_at(current_view: "ViewPort", idx: int, view: "ViewPort") -> "ViewPort":
    child = current_view.children[idx]
    view_embed_into(current_view, child)
    current_view.update_item_at(idx, view)
    current_view.update_size_new_root()
    return current_view


def view_embed_as_relative(
    current_view: "ViewPort", idx: int, into: "ViewPort"
) -> "ViewPort":
    into.relative_to(current_view)
    # TODO high priority double check this
    # view_embed_into(into, current_view[idx]) or view_embed_into(current_view[idx], into)
    current_view[idx] = view_embed_into(into, current_view[idx])
    current_view.update_size_new_root()
    return current_view


# graphics objects


@dataclass
class GOEmbedData:
    graphics_obj: GraphicsObject
    view: "ViewPort"


def go_embed_start_stop(data: GOEmbedData) -> GraphicsObject:
    if not isinstance(data.graphics_obj, (GOLine, GOAxis)):
        raise GGException("unexpecteda type")

    obj = data.graphics_obj
    obj.data.start = obj.data.start.embed_into(data.view)
    obj.data.stop = obj.data.stop.embed_into(data.view)
    return obj


def go_embed_rect(data: GOEmbedData) -> GraphicsObject:
    obj = cast(GORect, data.graphics_obj)
    obj.origin = obj.origin.embed_into(data.view)
    obj.width = obj.width.embed_into(AxisKind.X, data.view)
    obj.height = obj.height.embed_into(AxisKind.Y, data.view)
    return obj


def go_embed_raster(data: GOEmbedData) -> GraphicsObject:
    obj = cast(GORaster, data.graphics_obj)
    obj.origin = obj.origin.to_relative()
    obj.pixel_width = obj.pixel_width.to_relative_from_view(data.view, AxisKind.X)
    obj.pixel_height = obj.pixel_height.to_relative_from_view(data.view, AxisKind.Y)
    return obj


def go_embed_point(data: GOEmbedData) -> GraphicsObject:
    obj = cast(GOPoint, data.graphics_obj)
    obj.pos = obj.pos.embed_into(data.view)
    return obj


def go_embed_many_points(data: GOEmbedData) -> GraphicsObject:
    obj = cast(GOManyPoints, data.graphics_obj)
    obj.pos = [i.embed_into(data.view) for i in obj.pos]
    return obj


def go_embed_poly_line(data: GOEmbedData) -> GraphicsObject:
    obj = cast(GOPolyLine, data.graphics_obj)
    obj.pos = [i.embed_into(data.view) for i in obj.pos]
    return obj


def go_embed_text(data: GOEmbedData) -> GraphicsObject:
    if not isinstance(data.graphics_obj, (GOText, GOLabel, GOTickLabel)):
        raise GGException("unexpected")

    obj = data.graphics_obj
    obj.data.pos = obj.data.pos.embed_into(data.view)
    return obj


def go_embed_tick(data: GOEmbedData) -> GraphicsObject:
    obj = cast(GOTick, data.graphics_obj)
    obj.pos = obj.pos.embed_into(data.view)
    return obj


def go_embed_grid(data: GOEmbedData) -> GraphicsObject:
    obj = cast(GOGrid, data.graphics_obj)
    obj.origin = deepcopy(data.view.origin)
    obj.origin_diagonal = Coord(
        x=data.view.origin.x + RelativeCoordType(data.view.get_width().val),
        y=data.view.origin.y + RelativeCoordType(data.view.get_height().val),
    )

    obj.x_pos = [
        data.view.origin.x + i + RelativeCoordType(data.view.get_width().val)
        for i in obj.x_pos
    ]
    obj.y_pos = [
        data.view.origin.y + i + RelativeCoordType(data.view.get_height().val)
        for i in obj.y_pos
    ]

    return obj


def go_embed_composite(data: GOEmbedData) -> GraphicsObject:
    return data.graphics_obj


go_embed_lookup = {
    # start/stop
    GOType.AXIS: go_embed_start_stop,
    GOType.LINE: go_embed_start_stop,
    # text
    GOType.TEXT: go_embed_text,
    GOType.LABEL: go_embed_text,
    GOType.TICK_LABEL: go_embed_text,
    # others
    GOType.GRID_DATA: go_embed_grid,
    GOType.TICK_DATA: go_embed_tick,
    GOType.POINT_DATA: go_embed_point,
    GOType.MANY_POINTS_DATA: go_embed_many_points,
    GOType.POLYLINE_DATA: go_embed_poly_line,
    GOType.RECT_DATA: go_embed_rect,
    GOType.RASTER_DATA: go_embed_raster,
    GOType.COMPOSITE_DATA: go_embed_composite,
}


def graphics_object_embed_into(
    graphics_obj: GraphicsObject,
    view: "ViewPort",
) -> GraphicsObject:
    # TODO lot of this logic can be re used with go convert
    func = go_embed_lookup.get(graphics_obj.go_type)

    if not func:
        raise GGException("Conversion not possible")

    data = GOEmbedData(graphics_obj=graphics_obj, view=view)
    return func(data)
