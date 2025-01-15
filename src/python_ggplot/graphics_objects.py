from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, TypeVar, TYPE_CHECKING

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
from python_ggplot.units import Quantity

if TYPE_CHECKING:
    from python_ggplot.views import ViewPort


@dataclass
class GraphicsObjectConfig:
    style: Optional[Style]
    rotate_in_view: Optional[tuple[float, tuple[float, float]]] = None
    rotate: Optional[float] = None
    children: Optional[List["GraphicsObject"]] = None


class GOType(Enum):
    START_STOP = auto()
    TEXT = auto()
    GRID_DATA = auto()
    TICK_DATA = auto()
    POINT_DATA = auto()
    MANY_POINTS_DATA = auto()
    POLYLINE_DATA = auto()
    RECT_DATA = auto()
    RASTER_DATA = auto()
    COMPOSITE_DATA = auto()


@dataclass
class GraphicsObject:
    name: str
    config: GraphicsObjectConfig
    go_type: GOType

    def to_relative(
        self, view: Optional["ViewPort"] = None, axis: Optional[AxisKind] = None
    ) -> "GraphicsObject":
        from python_ggplot.graphics_object_convert import (
            graphics_object_to_relative,
        )  # pylint: disable=all

        return graphics_object_to_relative(self, view=view, axis=axis)


@dataclass
class StartStopData(GraphicsObject):
    start: Coord
    stop: Coord

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.START_STOP
        super().__init__(*args, **kwargs)


@dataclass
class TextData(GraphicsObject):
    text: str
    font: Font
    pos: Coord
    align: TextAlignKind

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.TEXT
        super().__init__(*args, **kwargs)


@dataclass
class RectData(GraphicsObject):
    origin: Coord
    width: Quantity
    height: Quantity

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.RECT_DATA
        super().__init__(*args, **kwargs)


@dataclass
class GridData(GraphicsObject):
    origin: Coord  # todo is this optinal? check later
    origin_diagonal: Coord
    x_pos: List[Coord1D]
    y_pos: List[Coord1D]

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.GRID_DATA
        super().__init__(*args, **kwargs)


@dataclass
class TickData(GraphicsObject):

    major: bool
    pos: Coord
    axis: AxisKind
    kind: TickKind
    secondary: bool

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.TICK_DATA
        super().__init__(*args, **kwargs)


@dataclass
class PointData(GraphicsObject):
    marker: MarkerKind
    pos: Coord
    size: float
    color: Color

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.POINT_DATA
        super().__init__(*args, **kwargs)


@dataclass
class ManyPointsData(GraphicsObject):

    marker: MarkerKind
    pos: List[Coord]
    size: float
    color: Color

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.MANY_POINTS_DATA
        super().__init__(*args, **kwargs)


@dataclass
class PolyLineData(GraphicsObject):
    pos: List[Coord]

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.POLYLINE_DATA
        super().__init__(*args, **kwargs)


@dataclass
class RasterData(GraphicsObject):
    origin: Coord
    pixel_width: Quantity
    pixel_height: Quantity
    block_x: int
    block_y: int
    draw_cb: Callable[[], List[int]]

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.RASTER_DATA
        super().__init__(*args, **kwargs)


@dataclass
class CompositeData(GraphicsObject):
    kind: CompositeKind

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.COMPOSITE_DATA
        super().__init__(*args, **kwargs)


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
