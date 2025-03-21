from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from python_ggplot.core.coord.objects import (
    Coord,
    Coord1D,
    CoordsInput,
    LengthCoord,
    PointCoordType,
    RelativeCoordType,
    path_coord_view_port,
)
from python_ggplot.core.objects import (
    BLACK,
    AxisKind,
    Color,
    GGException,
    Scale,
    Style,
    UnitType,
)
from python_ggplot.core.units.objects import Quantity, RelativeUnit
from python_ggplot.graphics.objects import GOType, GraphicsObject


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
    h_parent_view: Optional["Quantity"] = None
    w_parent_view: Optional["Quantity"] = None

    def __post_init__(self):
        self.update_views()

    def __setattr__(self, name: str, value: Any):
        super().__setattr__(name, value)
        if name in ("h_parent_view", "w_parent_view"):
            # prefer to have this here guaranteed,
            # this can cause weird bugs if not called manually
            self.update_views()

    def update_views(self):
        self.h_view = self.h_img
        self.w_view = self.w_img

        if self.w_parent_view is not None and self.h_parent_view is not None:
            if {self.w_parent_view.unit_type, self.h_parent_view.unit_type} != {
                UnitType.POINT
            }:
                raise GGException("parent view must have a point unit")
            self.h_view = self.h_parent_view
            self.w_view = self.w_parent_view


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

    def get_child_by_name(self, view_name: str) -> "ViewPort":
        """
        base layout:
            view.children[0].name = "top_left"
            view.children[1].name = "title"
            view.children[2].name = "top_right"
            view.children[3].name = "y_label"
            view.children[4].name = "plot"
            view.children[5].name = "legend" if theme_layout.requires_legend else "no_legend"
            view.children[6].name = "bottom_left"
            view.children[7].name = "x_label"
            view.children[8].name = "bottom_right"
        """
        # TODO replace view.children[4] with view.get_child_by_name("plot") etc
        for child in self.children:
            if child.name == view_name:
                return child

        raise GGException(f"View with name {view_name} not found")

    def get_coords(self) -> Dict[Any, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "origin": self.origin,
            "width": self.width,
            "height": self.height,
            "w_img": self.w_img,
            "h_img": self.h_img,
            "w_view": self.w_view,
            "h_view": self.h_view,
            "objects": [i.get_coords() for i in self.objects],
            "children": [i.get_coords() for i in self.children],
        }

    def gather_coords(self) -> Dict[Any, Any]:
        return self.get_coords()

    def find(self, go_type: GOType, recursive: bool = True):
        """
        TODO write docs and make this more flexible
        currently goes through all the object of current view and sub views
        returns the ones that match the type given
        """
        result: List[GraphicsObject] = []
        for object in self.objects:
            if object.go_type == go_type:
                result.append(object)

        if recursive:
            for sub_view in self.children:
                result.extend(sub_view.find(go_type))

        return result

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
            w_view=input_data.w_view,
            h_view=input_data.h_view,
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

        input_data.h_img = self.h_img.to_points()
        input_data.w_img = self.w_img.to_points()

        input_data.h_parent_view = self.point_height()
        input_data.w_parent_view = self.point_width()

        result = ViewPort.from_input(origin, width, height, input_data)
        return result

    @staticmethod
    def from_coords(coords_input: CoordsInput, view_input: ViewPortInput) -> "ViewPort":
        origin = Coord(
            x=RelativeCoordType(coords_input.left),
            y=RelativeCoordType(coords_input.bottom),
        )
        width = RelativeUnit(coords_input.width)
        height = RelativeUnit(coords_input.height)
        result = ViewPort.from_input(origin, width, height, view_input)
        return result

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

    def __getitem__(self, k: int) -> "ViewPort":
        return self.children[k]

    def __setitem__(self, k: int, v: "ViewPort"):
        self.children[k] = v

    def len(self) -> int:
        return len(self.children)

    def high(self) -> int:
        return self.len() - 1

    def get_child(self, idx: int) -> "ViewPort":
        return self.children[idx]

    def to_relative_dimension(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            return self.get_width()
        if axis_kind == AxisKind.Y:
            return self.get_height()
        raise GGException("unexpected")

    def embed_as_relative(self: "ViewPort", idx: int, into: "ViewPort") -> "ViewPort":
        from python_ggplot.core.embed import (
            view_embed_as_relative,
        )  # pylint: disable=all

        return view_embed_as_relative(self, idx, into)

    def embed_into(self: "ViewPort", into: "ViewPort") -> "ViewPort":
        from python_ggplot.core.embed import view_embed_into  # pylint: disable=all

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

    def update_item_at(self, idx: int, view: "ViewPort"):
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

    def x_axis(self, width: float = 1.0, color: Color = BLACK) -> GraphicsObject:
        from python_ggplot.graphics.initialize import InitAxisInput, init_axis

        return init_axis(AxisKind.X, InitAxisInput(width=width, color=color))

    def y_axis(self, width: float = 1.0, color: Color = BLACK) -> GraphicsObject:
        from python_ggplot.graphics.initialize import InitAxisInput, init_axis

        return init_axis(AxisKind.Y, InitAxisInput(width=width, color=color))

    # def __rich_repr__(self):
    #     yield "width", self.width
    #     yield "height", self.height
    #     yield "w_img", self.w_img
    #     yield "h_img", self.h_img
    #     yield "w_view", self.w_view
    #     yield "h_view", self.h_view


def x_axis_y_pos(
    view: Optional[ViewPort] = None,
    margin: float = 0.0,
    is_secondary: bool = False,
) -> Coord1D:
    if view:
        pos = -margin if is_secondary else view.point_height().val + margin
        length = view.point_height() if is_secondary else view.h_img
        result = PointCoordType(pos, LengthCoord(length=length))
        return result
    else:
        pos = 0.0 if is_secondary else 1.0
        return RelativeCoordType(pos)


def y_axis_x_pos(
    view: Optional[ViewPort] = None,
    margin: float = 0.0,
    is_secondary: bool = False,
) -> Coord1D:
    if view:
        pos = view.point_width().val + margin if is_secondary else -margin
        length = view.w_img if is_secondary else view.point_width()
        result = PointCoordType(pos, LengthCoord(length=length))
        return result
    else:
        pos = 1.0 if is_secondary else 0.0
        return RelativeCoordType(pos)


# TODO This is old logic and probably need to be deleted,
# but keeping for now until i double check why it was there
# def x_axis_y_pos(
#     view: Optional[ViewPort] = None,
#     margin: float = 0.0,
#     is_secondary: bool = False,
# ) -> Coord1D:
#     if view:
#         pos = view.height.val + margin if is_secondary else -margin
#         coord = quantitiy_to_coord(view.height, pos)
#         coord.pos = pos
#         return coord
#     else:
#         pos = 0.0 if is_secondary else 1.0
#         return RelativeCoordType(pos)


# def y_axis_x_pos(
#     view: Optional[ViewPort] = None,
#     margin: float = 0.0,
#     is_secondary: bool = False,
# ) -> Coord1D:
#     if view:
#         pos = view.width.val + margin if is_secondary else -margin
#         coord = quantitiy_to_coord(view.width, pos)
#         coord.pos = pos
#         return coord
#     else:
#         pos = 1.0 if is_secondary else 0.0
#         return RelativeCoordType(pos)
