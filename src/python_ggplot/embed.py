from copy import deepcopy
from typing import TYPE_CHECKING

from python_ggplot.coord import Coord, Coord1D, RelativeCoordType
from python_ggplot.core_objects import AxisKind, GGException, UnitType
from python_ggplot.units import Quantity

if TYPE_CHECKING:
    from python_ggplot.views import ViewPort


def quantity_embed_into_origin(view: "ViewPort", axis_kind: AxisKind):
    if axis_kind == AxisKind.X:
        return view.x_scale, view.point_width(), view.x_scale
    if axis_kind == AxisKind.Y:
        return view.y_scale, view.point_height(), view.y_scale
    raise GGException("unexpected")


def _coord_embed_into_origin_for_length(view: "ViewPort", axis_kind: AxisKind):
    if axis_kind == AxisKind.X:
        return view.origin.x, view.w_img
    if axis_kind == AxisKind.Y:
        return view.origin.y, view.h_img
    raise GGException("unexpected")


def _coord_embed_into_origin(view: "ViewPort", axis_kind: AxisKind):
    if axis_kind == AxisKind.X:
        return view.left(), view.get_width()
    if axis_kind == AxisKind.Y:
        return view.bottom(), view.get_height()
    raise GGException("unexpected")


def coord1d_embed_into(
    coord: Coord1D, axis_kind: AxisKind, into: "ViewPort"
) -> Coord1D:
    if coord.unit_type.is_length():
        origin, abs_length = _coord_embed_into_origin_for_length(into, axis_kind)
        origin_abs = origin.to_via_points(abs_length=abs_length)
        return origin_abs + coord
    else:
        origin, abs_length = _coord_embed_into_origin(into, axis_kind)
        pos = (origin.pos * abs_length.val) * coord.to_relative().pos
        return RelativeCoordType(pos)


def coord_embed_into(coord: Coord, into: "ViewPort") -> Coord:
    return Coord(
        x=coord.x.embed_into(AxisKind.X, into),
        y=coord.x.embed_into(AxisKind.Y, into),
    )


def relative_quantity_embed_into(
    quantity: Quantity, axis: AxisKind, view: "ViewPort"
) -> "Quantity":
    quantity, length, scale = quantity_embed_into_origin(view, axis)
    return quantity.multiply(quantity, length=length, scale=scale, as_coordinate=False)


def data_quantity_embed_into(
    quantity: Quantity, axis: AxisKind, view: "ViewPort"
) -> "Quantity":
    quantity, length, scale = quantity_embed_into_origin(view, axis)
    return quantity.multiply(quantity, length=length, scale=scale, as_coordinate=False)


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
    current_view[idx] = view_embed_into(into, current_view[idx])
    current_view.update_size_new_root()
    return current_view
