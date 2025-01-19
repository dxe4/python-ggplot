from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from python_ggplot.core.coord.objects import (
    Coord,
    Coord1D,
    CoordsInput,
    RelativeCoordType,
    path_coord_view_port,
)
from python_ggplot.core.objects import AxisKind, GGException, Scale, Style, UnitType
from python_ggplot.core.units.convert import quantitiy_to_coord
from python_ggplot.core.units.objects import PointUnit, Quantity, RelativeUnit
from python_ggplot.graphics.objects import GraphicsObject


@dataclass
class ViewPortInput:
    name: str = ""
    parent: str = ""
    w_img: "Quantity" = field(default_factory=lambda: Quantity.points(640.0))
    h_img: "Quantity" = field(default_factory=lambda: Quantity.points(480.0))
    style: Optional["Style"] = None
    x_scale: Optional["Scale"] = None
    y_scale: Optional["Scale"] = None
    rotate: Optional[float] = None
    scale: Optional[float] = None
    objects: List["GraphicsObject"] = field(default_factory=list)
    children: List["ViewPort"] = field(default_factory=list)
    w_view: Optional["Quantity"] = None
    h_view: Optional["Quantity"] = None

    def update_from_viewport(self, view: "ViewPort"):
        self.h_img = view.h_img.to_points()
        self.w_img = view.w_img.to_points()

        self.h_view = view.point_height()
        self.w_view = view.point_width()

    @staticmethod
    def get_views(
        w_view_quantity: Optional[Quantity] = None,
        h_view_quantity: Optional[Quantity] = None,
    ) -> Tuple["Quantity", "Quantity"]:
        if w_view_quantity is not None and h_view_quantity is not None:
            if {w_view_quantity.unit_type, h_view_quantity.unit_type} != {
                UnitType.POINT
            }:
                raise GGException("parent view must have a point unit")
            return (deepcopy(w_view_quantity), deepcopy(h_view_quantity))
        return (PointUnit(640.0), PointUnit(480.0))

    @staticmethod
    def new(
        w_view_quantity: Optional[Quantity] = None,
        h_view_quantity: Optional[Quantity] = None,
    ) -> "ViewPortInput":
        w_view, h_view = ViewPortInput.get_views(w_view_quantity, h_view_quantity)
        return ViewPortInput(
            w_view=w_view,
            h_view=h_view,
            h_img=PointUnit(h_view.val),
            w_img=PointUnit(w_view.val),
        )


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

    def get_center(self) -> Tuple[float, float]:
        center_x: float = self.left().pos + (
            self.get_width().to_relative(length=self.point_width()).val / 2.0
        )

        center_y: float = self.bottom().pos + (
            self.get_height().to_relative(length=self.point_height()).val / 2.0
        )

        return center_x, center_y

    def scale_for_axis(self, axis: AxisKind) -> Optional[Scale]:
        if axis == AxisKind.X:
            return self.x_scale
        if axis == AxisKind.Y:
            return self.y_scale
        raise GGException("unexpected")

    def add_obj(self, obj: GraphicsObject):
        if obj.config.style is None:
            obj.config.style = self.style

        self.objects.append(obj)

    @staticmethod
    def from_input(
        origin: Coord, width: Quantity, height: Quantity, input_data: ViewPortInput
    ) -> "ViewPort":
        w_view, h_view = ViewPortInput.get_views(input_data.w_view, input_data.h_view)

        result = ViewPort(
            origin=origin,
            width=width,
            height=height,
            name=input_data.name,
            parent=input_data.parent,
            style=input_data.style,
            x_scale=input_data.x_scale,
            y_scale=input_data.y_scale,
            rotate=input_data.rotate,
            scale=input_data.scale,
            w_view=w_view,
            h_view=h_view,
            w_img=input_data.w_img,
            h_img=input_data.h_img,
        )
        return result

    def add_viewport(
        self,
        origin: Coord,
        width: Quantity,
        height: Quantity,
        input_data: ViewPortInput,
    ) -> "ViewPort":
        origin = path_coord_view_port(origin, self)

        input_data.update_from_viewport(self)

        return ViewPort.from_input(origin, width, height, input_data)

    @staticmethod
    def from_coords(coords_input: CoordsInput, view_input: ViewPortInput) -> "ViewPort":
        origin = Coord(
            x=RelativeCoordType(coords_input.left),
            y=RelativeCoordType(coords_input.bottom),
        )
        width = RelativeUnit(coords_input.width)
        height = RelativeUnit(coords_input.height)
        return ViewPort.from_input(origin, width, height, view_input)

    def add_viewport_from_coords(
        self, coords_input: CoordsInput, input_data: ViewPortInput
    ):
        origin = Coord(
            x=RelativeCoordType(coords_input.left),
            y=RelativeCoordType(coords_input.bottom),
        )
        width = RelativeUnit(coords_input.width)
        height = RelativeUnit(coords_input.height)

        input_data.x_scale = self.x_scale or input_data.x_scale
        input_data.y_scale = self.y_scale or input_data.y_scale

        return self.add_viewport(origin, width, height, input_data)

    def update_data_scale(self):
        self.update_data_scale_for_objects(self.objects)
        for child in self.children:
            child.x_scale = self.x_scale
            child.y_scale = self.y_scale
            child.update_data_scale()

    def update_data_scale_for_objects(self, objects: List[GraphicsObject]):
        for obj in objects:
            self.update_data_scale_for_object(obj)

    def update_data_scale_for_object(self, obj: GraphicsObject):
        obj.update_view_scale(self)

    def get_width(self) -> Quantity:
        return self.width.to_relative(length=self.w_img)

    def get_height(self):
        return self.height.to_relative(length=self.h_img)

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

    def embed_as_relative(self: "ViewPort", idx: int, into: "ViewPort") -> "ViewPort":
        from python_ggplot.embed import view_embed_as_relative  # pylint: disable=all

        return view_embed_as_relative(self, idx, into)

    def embed_into(self: "ViewPort", into: "ViewPort") -> "ViewPort":
        from python_ggplot.embed import view_embed_into  # pylint: disable=all

        return view_embed_into(self, into)

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

    def length_from_axis(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            return self.point_width()
        if axis_kind == AxisKind.Y:
            return self.point_height()
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

    def point_width(self) -> "Quantity":
        if not self.w_view:
            raise GGException("expected w view")

        if self.w_view.unit_type != UnitType.POINT:
            raise GGException(f"Expected Point, found {self.w_view.unit_type}")

        other = self.width.to_relative(length=self.w_view)
        result = self.w_view.multiply(other, length=self.w_view)
        return result

    def point_height(self) -> "Quantity":
        if not self.h_view:
            raise GGException("expected w view")

        if self.h_view.unit_type != UnitType.POINT:
            raise GGException(f"Expected Point, found {self.h_view.unit_type}")

        other = self.height.to_relative(length=self.h_view)
        result = self.h_view.multiply(other, length=self.h_view)
        return result

    # def __rich_repr__(self):
    #     yield "width", self.width
    #     yield "height", self.height
    #     yield "w_img", self.w_img
    #     yield "h_img", self.h_img
    #     yield "w_view", self.w_view
    #     yield "h_view", self.h_view


def x_axis_y_pos(
    view: Optional[ViewPort] = None,
    margin: Optional[float] = 0.0,
    is_secondary: Optional[bool] = False,
) -> Coord1D:
    is_secondary = is_secondary if is_secondary is not None else False
    margin = margin if margin is not None else 0.0

    if view:
        pos = view.height.val + margin if is_secondary else -margin
        coord = quantitiy_to_coord(view.height, pos)
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
        pos = viewport.width.val + margin if is_secondary else -margin
        coord = quantitiy_to_coord(viewport.width, pos)
        coord.pos = pos
        return coord
    else:
        pos = 1.0 if is_secondary else 0.0
        return RelativeCoordType(pos)
