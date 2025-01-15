from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

from python_ggplot.coord import Coord, Coord1D, RelativeCoordType
from python_ggplot.core_objects import AxisKind, GGException, Scale, Style, UnitType
from python_ggplot.graphics_objects import GraphicsObject
from python_ggplot.quantity_convert import quantitiy_to_coord
from python_ggplot.units import Quantity


@dataclass
class ViewPort:
    origin: Coord
    width: Quantity
    height: Quantity

    w_img: Quantity
    h_img: Quantity

    name: str
    parent: str

    style: Optional[Style]
    x_scale: Optional[Scale]
    y_scale: Optional[Scale]

    rotate: Optional[float] = None
    scale: Optional[float] = None

    objects: List[GraphicsObject] = field(default_factory=list)
    children: List["ViewPort"] = field(default_factory=list)

    w_view: Optional[Quantity] = None
    h_view: Optional[Quantity] = None

    def __getitem__(self, k):
        return self.children[k]

    def __setitem__(self, k, v):
        self.children[k] = v

    def len(self) -> int:
        return len(self.children)

    def high(self) -> int:
        return self.len() - 1

    def get_child(self, idx) -> "ViewPort":
        return self.children[idx]

    def to_relative_dimension(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            return self.get_width()
        if axis_kind == AxisKind.Y:
            return self.get_height()
        raise GGException("unexpected")

    def relative_to(self, other: "ViewPort"):
        origin = self.origin.to_relative()
        width = self.width.to_relative_from_view(other, AxisKind.X)
        height = self.height.to_relative_from_view(other, AxisKind.Y)

        new_viewport = deepcopy(self)
        new_viewport.origin = origin
        new_viewport.width = width
        new_viewport.height = height

        for obj in self.objects:
            obj.to_relative(other)

        for child in self.children:
            child.relative_to(new_viewport)

    def left(self):
        return self.origin.x.to_relative(None)

    def bottom(self):
        return self.origin.y.to_relative(None)

    def get_width(self):
        return self.height.to_relative(length=self.w_img)

    def get_height(self):
        return self.height.to_relative(length=self.h_img)

    def length_from_axis(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            return self.point_width()
        if axis_kind == AxisKind.Y:
            return self.point_height()
        raise GGException("unexpected")

    def scale_for_axis(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            return self.x_scale
        if axis_kind == AxisKind.Y:
            return self.y_scale
        raise GGException("unexpected")

    def update_scale_1d(self, coord: Coord1D):
        coord.update_scale(self)

    def update_scale(self, coord: Coord):
        self.update_scale_1d(coord.x)
        self.update_scale_1d(coord.y)

    def update_size_new_root(self):
        for child in self.children:
            child.w_img = deepcopy(self.w_img)
            child.h_img = deepcopy(self.h_img)
            child.update_size_new_root()

    def update_item_at(self, idx, view: "ViewPort"):
        self.children[idx] = deepcopy(view)

    # def apply_operator(
    #     self,
    #     other: Quantity,
    #     length: Optional[Quantity],
    #     scale: Optional[Scale],
    #     as_coordinate: bool,
    #     operator: Callable[[float, float], float],
    # ) -> "Quantity":
    #     pass

    def point_width_height(self, dimension: Optional["Quantity"]) -> "Quantity":
        if not self.w_view:
            raise GGException("expected w view")

        if self.w_view.unit_type != UnitType.POINT:
            raise ValueError(f"Expected Point, found {self.w_view.unit_type}")

        other = self.width.to_relative(dimension)
        return self.w_view.multiply(other)

    def point_width(self) -> "Quantity":
        return self.point_width_height(self.w_view)

    def point_height(self) -> "Quantity":
        return self.point_width_height(self.h_view)


def x_axis_y_pos(
    viewport: Optional[ViewPort] = None,
    margin: Optional[float] = 0.0,
    is_secondary: Optional[bool] = False,
) -> Coord1D:
    is_secondary = is_secondary if is_secondary is not None else False
    margin = margin if margin is not None else 0.0

    if viewport:
        coord = quantitiy_to_coord(viewport.height)
        pos = viewport.height.val + margin if is_secondary else -margin
        coord.pos = pos
        return coord
    else:
        pos = 0.0 if is_secondary else 1.0
        return RelativeCoordType(pos)


def y_axis_x_pos(
    viewport: Optional[ViewPort] = None,
    margin: Optional[float] = 0.0,
    is_secondary: Optional[bool] = False,
) -> Coord1D:
    is_secondary = is_secondary if is_secondary is not None else False
    margin = margin if margin is not None else 0.0

    if viewport:
        coord = quantitiy_to_coord(viewport.width)
        pos = viewport.width.val + margin if is_secondary else -margin
        coord.pos = pos
        return coord
    else:
        pos = 1.0 if is_secondary else 0.0
        return RelativeCoordType(pos)
