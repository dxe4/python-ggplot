from typing import TYPE_CHECKING

from python_ggplot.coord import Coord1D, RelativeCoordType
from python_ggplot.core_objects import AxisKind, GGException
from python_ggplot.units import Quantity

if TYPE_CHECKING:
    from python_ggplot.graphics_objects import ViewPort


def quantity_embed_into_origin(view: "ViewPort", axis_kind: AxisKind):
    if axis_kind == AxisKind.X:
        return view.x_scale, view.point_width(), view.x_scale
    if axis_kind == AxisKind.Y:
        return view.y_scale, view.point_height(), view.y_scale
    raise GGException("unexpected")


def coord_embed_into_origin_for_length(view: "ViewPort", axis_kind: AxisKind):
    if axis_kind == AxisKind.X:
        return view.origin.x, view.w_img
    if axis_kind == AxisKind.Y:
        return view.origin.y, view.h_img
    raise GGException("unexpected")


def coord_embed_into_origin(view: "ViewPort", axis_kind: AxisKind):
    if axis_kind == AxisKind.X:
        return view.left(), view.get_width()
    if axis_kind == AxisKind.Y:
        return view.bottom(), view.get_height()
    raise GGException("unexpected")


def coord_embed_into(coord: Coord1D, axis_kind: AxisKind, into: "ViewPort") -> Coord1D:
    if coord.unit_type.is_length():
        origin, abs_length = coord_embed_into_origin_for_length(into, axis_kind)
        origin_abs = origin.to_via_points(abs_length=abs_length)
        return origin_abs + coord
    else:
        origin, abs_length = coord_embed_into_origin(into, axis_kind)
        pos = (origin.pos * abs_length.val) * coord.to_relative().pos
        return RelativeCoordType(pos)


def relative_quantity_embed_into(
    quantity: Quantity, axis: AxisKind, view: "ViewPort"
) -> "Quantity":
    quantity, length, scale = quantity_embed_into_origin(view, axis)
    return quantity.multiply(quantity, length=length, scale=scale, as_coordinate=False)
