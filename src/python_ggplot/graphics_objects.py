from dataclasses import dataclass
from typing import Callable, List, Optional
import typing as tp

from python_ggplot.coord import Coord, Coord1D, RelativeCoordType, quantitiy_to_coord
from python_ggplot.core_objects import (
    AxisKind,
    Color,
    CompositeKind,
    Font,
    GGException,
    MarkerKind,
    TextAlignKind,
    TickKind,
    Style,
    Scale
)
from python_ggplot.units import Quantity, ToQuantityData


class GraphObjectKind:
    pass


@dataclass
class StartStopData(GraphObjectKind):
    start: Coord
    stop: Coord


@dataclass
class TextData(GraphObjectKind):
    text: str
    font: Font
    pos: Coord
    align: TextAlignKind


@dataclass
class GridData(GraphObjectKind):
    origin: Coord
    origin_diagonal: Coord
    x_post: List[Coord1D]
    y_post: List[Coord1D]


@dataclass
class TickData(GraphObjectKind):
    major: bool
    pos: Coord
    axis: AxisKind
    kind: TickKind
    secondary: bool


@dataclass
class PointData(GraphObjectKind):
    marker: MarkerKind
    pos: Coord
    size: float
    color: Color


@dataclass
class ManyPointsData(GraphObjectKind):
    marker: MarkerKind
    pos: List[Coord]
    size: float
    color: Color


@dataclass
class PolyLineData(GraphObjectKind):
    pos: List[Coord]


@dataclass
class RectData(GraphObjectKind):
    origin: Coord
    width: Quantity
    height: Quantity


@dataclass
class RasterData(GraphObjectKind):
    origin: Coord
    pixel_width: Quantity
    pixel_height: Quantity
    block_x: int
    block_y: int
    draw_cb: Callable[[], List[int]]


@dataclass
class CompositeData(GraphObjectKind):
    kind: CompositeKind


@dataclass
class GraphicsObject:
    name: str
    style: Optional[Style]
    rotate_in_view: Optional[tuple[float, tuple[float, float]]] = None
    rotate: Optional[float] = None
    children: List["GraphicsObject"]
    graphics_kind: Optional[GraphObjectKind] = None


@dataclass
class ViewPort:
    name: str
    parent: str
    style: Style
    x_scale: Scale
    y_scale: Scale
    rotate: Optional[float] = None
    scale: Optional[float] = None
    origin: Coord
    width: Quantity
    height: Quantity
    objects: List[GraphicsObject]
    children: List["ViewPort"]
    w_view: Quantity
    h_view: Quantity
    w_img: Quantity
    h_img: Quantity

    def left(self):
        return self.origin.x.to_relative(None)

    def get_width(self):
        return self.height.to_relative(ToQuantityData(length=self.w_img))

    def bottom(self):
        return self.origin.y.to_relative(None)

    def get_height(self):
        return self.height.to_relative(ToQuantityData(length=self.h_img))

    def embed_into_origin_for_length(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            return self.origin.x, self.w_img
        if axis_kind == AxisKind.Y:
            return self.origin.y, self.h_img
        raise GGException("unexpected")

    def embed_into_origin(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            return self.left(), self.get_width()
        if axis_kind == AxisKind.Y:
            return self.bottom(), self.get_height()
        raise GGException("unexpected")

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

        if not self.w_view.unit.is_point():
            raise ValueError(f"Expected Point, found {self.w_view.unit}")

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
