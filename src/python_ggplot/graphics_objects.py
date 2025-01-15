from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from python_ggplot.coord import Coord, Coord1D, RelativeCoordType
from python_ggplot.core_objects import (
    AxisKind,
    Color,
    CompositeKind,
    Font,
    GGException,
    MarkerKind,
    Scale,
    Style,
    TextAlignKind,
    TickKind,
    UnitType,
)
from python_ggplot.quantity_convert import quantitiy_to_coord
from python_ggplot.units import PointUnit, Quantity


@dataclass
class GraphicsObjectConfig:
    style: Optional[Style]
    rotate_in_view: Optional[tuple[float, tuple[float, float]]] = None
    rotate: Optional[float] = None
    children: Optional[List["GraphicsObject"]] = None


@dataclass
class GraphicsObject:
    name: str
    config: GraphicsObjectConfig


class GOType(Enum):
    START_STOP = auto()
    TEXT = auto()
    GRID_DATA = auto()
    TICK_DATA = auto()
    POINT_DATA = auto()
    MANY_POINTS_DATA = auto()
    POLYLINE_DATA = auto()
    REC_DATA = auto()
    RASTER_DATA = auto()
    COMPOSITE_DATA = auto()


@dataclass
class StartStopData(GraphicsObject):
    go_type = GOType.START_STOP
    start: Coord
    stop: Coord

    def to_relative(self):
        self.start = self.start.to_relative()
        self.stop = self.stop.to_relative()


@dataclass
class TextData(GraphicsObject):
    go_type = GOType.TEXT
    text: str
    font: Font
    pos: Coord
    align: TextAlignKind

    # def to_relative(self):
    #     self.pos = self.pos.to_relative()

    # def to_relative_with_view(self, view: "ViewPort"):
    #     self.origin = self.origin.to_relative()
    #     self.width = self.width.to_relative_with_view(view, AxisKind.X)
    #     self.height = self.height.to_relative_with_view(view, AxisKind.Y)


@dataclass
class GridData(GraphicsObject):
    go_type = GOType.GRID_DATA
    origin: Coord
    origin_diagonal: Coord
    x_post: List[Coord1D]
    y_post: List[Coord1D]


@dataclass
class TickData(GraphicsObject):
    go_type = GOType.TICK_DATA
    major: bool
    pos: Coord
    axis: AxisKind
    kind: TickKind
    secondary: bool


@dataclass
class PointData(GraphicsObject):
    go_type = GOType.POINT_DATA
    marker: MarkerKind
    pos: Coord
    size: float
    color: Color


@dataclass
class ManyPointsData(GraphicsObject):
    go_type = GOType.MANY_POINTS_DATA
    marker: MarkerKind
    pos: List[Coord]
    size: float
    color: Color


@dataclass
class PolyLineData(GraphicsObject):
    go_type = GOType.POLYLINE_DATA
    pos: List[Coord]


@dataclass
class RasterData(GraphicsObject):
    go_type = GOType.RASTER_DATA
    origin: Coord
    pixel_width: Quantity
    pixel_height: Quantity
    block_x: int
    block_y: int
    draw_cb: Callable[[], List[int]]


@dataclass
class CompositeData(GraphicsObject):
    go_type = GOType.COMPOSITE_DATA
    kind: CompositeKind


@dataclass
class ViewPortInput:
    name: str = ""
    parent: str = ""
    w_img: Optional["Quantity"] = field(default_factory=lambda: Quantity(640.0))
    h_img: Optional["Quantity"] = field(default_factory=lambda: Quantity(480.0))
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
        self.w_img = view.h_img.to_points()

        self.h_view = view.point_height()
        self.w_view = view.point_width()

    @staticmethod
    def get_views(
        w_view_quantity: Optional[Quantity] = None,
        h_view_quantity: Optional[Quantity] = None,
    ) -> Tuple["Quantity", "Quantity"]:
        if w_view_quantity is not None and h_view_quantity is not None:
            if {w_view_quantity.unit_type, h_view_quantity} != {UnitType.POINT}:
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
    name: str
    parent: str
    style: Style
    x_scale: Scale
    y_scale: Scale
    origin: Coord
    width: Quantity
    height: Quantity
    objects: List[GraphicsObject]
    children: List["ViewPort"]
    w_view: Quantity
    h_view: Quantity
    w_img: Quantity
    h_img: Quantity
    rotate: Optional[float] = None
    scale: Optional[float] = None

    def to_relative_dimension(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            return self.get_width()
        if axis_kind == AxisKind.Y:
            return self.get_height()
        raise GGException("unexpected")

    def left(self):
        return self.origin.x.to_relative(None)

    def get_width(self):
        return self.height.to_relative(length=self.w_img)

    def bottom(self):
        return self.origin.y.to_relative(None)

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

    def apply_operator(
        self,
        other: Quantity,
        length: Optional[Quantity],
        scale: Optional[Scale],
        as_coordinate: bool,
        operator: Callable[[float, float], float],
    ) -> "Quantity":
        pass

    def point_height_height(self, dimension: "Quantity") -> "Quantity":

        if self.w_view.unit_type != UnitType.POINT:
            raise ValueError(f"Expected Point, found {self.w_view.unit_type}")

        other = self.width.to_relative(dimension)
        return self.w_view.multiply(other)

    def point_width(self) -> "Quantity":
        return self.point_height_height(self.w_view)

    def point_height(self) -> "Quantity":
        return self.point_height_height(self.h_view)


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


def format_tick_value(f: float, scale: Optional[float] = None) -> str:
    scale = scale if scale is not None else 0.0
    tick_precision_cutoff = 6.0
    # tick_precision = 5.0

    if abs(f) < scale / 10.0:
        return "0"
    elif (
        abs(f) >= 10.0**tick_precision_cutoff
        or abs(f) <= 10.0**-tick_precision_cutoff
    ):
        return f"{f:5.3e}".rstrip("0")
    else:
        return f"{f:.5f}".rstrip("0")
