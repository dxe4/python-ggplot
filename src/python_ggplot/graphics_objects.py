from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, TypeVar

from python_ggplot.coord import Coord, Coord1D
from python_ggplot.core_objects import (
    AxisKind,
    Color,
    CompositeKind,
    Font,
    GGException,
    MarkerKind,
    Style,
    TextAlignKind,
    TickKind,
)
from python_ggplot.units import  Quantity


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

    def to_relative(self, other):
        # todo implemeent
        raise GGException("TODO impl")


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
class CoordsInput:
    left: float = 0.0
    bottom: float = 0.0
    width: float = 1.0
    height: float = 1.0


T = TypeVar("T")


def first_option(left: Optional[T], right: Optional[T]) -> Optional[T]:
    if left is not None:
        return left
    return right


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
