from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, TypeVar

from python_ggplot.core.coord.objects import Coord, Coord1D
from python_ggplot.core.objects import (
    BLACK,
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
from python_ggplot.core.units.objects import Quantity

if TYPE_CHECKING:
    from python_ggplot.graphics.views import ViewPort


@dataclass
class GraphicsObjectConfig:
    style: Optional[Style] = None
    rotate_in_view: Optional[tuple[float, tuple[float, float]]] = None
    rotate: Optional[float] = None
    children: Optional[List["GraphicsObject"]] = field(default_factory=list)


class GOType(Enum):
    # start/stop data
    LINE = auto()
    AXIS = auto()
    # text
    TEXT = auto()
    TICK_LABEL = auto()
    LABEL = auto()
    # others
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

    def get_pos(self):
        raise GGException("graphics object has no position")

    def update_view_scale(self, view: "ViewPort"):
        raise GGException("this should never reach")

    def embed_into(
        self, view: "ViewPort", axis: Optional[AxisKind] = None
    ) -> "GraphicsObject":
        from python_ggplot.embed import (
            graphics_object_to_relative,
        )  # pylint: disable=all

        return graphics_object_to_relative(self, view, axis)

    def to_relative(
        self, view: Optional["ViewPort"] = None, axis: Optional[AxisKind] = None
    ) -> "GraphicsObject":
        from python_ggplot.graphics.convert import (
            graphics_object_to_relative,
        )  # pylint: disable=all

        return graphics_object_to_relative(self, view=view, axis=axis)


@dataclass
class StartStopData:
    start: Coord
    stop: Coord


@dataclass
class GOAxis(GraphicsObject):
    data: StartStopData

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.start)
        view.update_scale(self.data.stop)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.AXIS
        super().__init__(*args, **kwargs)


@dataclass
class GOLine(GraphicsObject):
    data: StartStopData

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.start)
        view.update_scale(self.data.stop)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.LINE
        super().__init__(*args, **kwargs)


@dataclass
class TextData:
    text: str
    font: Font
    pos: Coord
    align: TextAlignKind


@dataclass
class GOText(GraphicsObject):
    data: TextData

    def get_pos(self):
        return self.data.pos

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.pos)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.TEXT
        super().__init__(*args, **kwargs)


@dataclass
class GOLabel(GraphicsObject):
    data: TextData

    def get_pos(self):
        return self.data.pos

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.pos)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.LABEL
        super().__init__(*args, **kwargs)


@dataclass
class GOTickLabel(GraphicsObject):
    data: TextData

    def get_pos(self):
        return self.data.pos

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.pos)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.TICK_LABEL
        super().__init__(*args, **kwargs)


@dataclass
class GORect(GraphicsObject):
    origin: Coord
    width: Quantity
    height: Quantity

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.origin)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.RECT_DATA
        super().__init__(*args, **kwargs)


@dataclass
class GOGrid(GraphicsObject):
    x_pos: List[Coord1D]
    y_pos: List[Coord1D]
    origin: Optional[Coord]
    origin_diagonal: Optional[Coord]

    def update_view_scale(self, view: "ViewPort"):
        for x_pos in self.x_pos:
            view.update_scale_1d(x_pos)
        for y_pos in self.y_pos:
            view.update_scale_1d(y_pos)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.GRID_DATA
        super().__init__(*args, **kwargs)


@dataclass
class GOTick(GraphicsObject):
    major: bool
    pos: Coord
    axis: AxisKind
    kind: TickKind
    secondary: bool

    def get_pos(self):
        return self.pos

    def scale_for_axis(self, axis: AxisKind) -> Scale:
        if axis == AxisKind.X:
            return self.pos.x.get_scale()
        if axis == AxisKind.Y:
            return self.pos.y.get_scale()
        raise GGException("unexpected")

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.TICK_DATA
        super().__init__(*args, **kwargs)


@dataclass
class GOPoint(GraphicsObject):
    marker: MarkerKind
    pos: Coord
    size: float
    color: Color

    def get_pos(self):
        return self.pos

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.POINT_DATA
        super().__init__(*args, **kwargs)


@dataclass
class GOManyPoints(GraphicsObject):
    marker: MarkerKind
    pos: List[Coord]
    size: float
    color: Color

    def update_view_scale(self, view: "ViewPort"):
        for pos in self.pos:
            view.update_scale(pos)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.MANY_POINTS_DATA
        super().__init__(*args, **kwargs)


@dataclass
class GOPolyLine(GraphicsObject):
    pos: List[Coord]

    def update_view_scale(self, view: "ViewPort"):
        for pos in self.pos:
            view.update_scale(pos)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.POLYLINE_DATA
        super().__init__(*args, **kwargs)


@dataclass
class GORaster(GraphicsObject):
    origin: Coord
    pixel_width: Quantity
    pixel_height: Quantity
    block_x: int
    block_y: int
    draw_cb: Callable[[], List[int]]

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.origin)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.RASTER_DATA
        super().__init__(*args, **kwargs)


@dataclass
class GOComposite(GraphicsObject):
    kind: CompositeKind

    def update_view_scale(self, view: "ViewPort"):
        if self.config.children is None:
            return

        for go in self.config.children:
            go.update_view_scale(view)

    def __init__(self, *args, **kwargs):
        kwargs["go_type"] = GOType.COMPOSITE_DATA
        super().__init__(*args, **kwargs)


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


def go_update_data_scale(go: GraphicsObject, view: "ViewPort"):
    go.update_view_scale(view)
