from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, TypeVar

from python_ggplot.core.coord.objects import Coord, Coord1D
from python_ggplot.core.objects import (
    BLACK,
    AxisKind,
    Color,
    CompositeKind,
    Font,
    GGException,
    Image,
    MarkerKind,
    Point,
    Scale,
    Style,
    TextAlignKind,
    TickKind,
    UnitType,
)
from python_ggplot.core.units.objects import Quantity

if TYPE_CHECKING:
    from python_ggplot.core.objects import Image
    from python_ggplot.graphics.views import ViewPort


def coord1d_to_abs_image(coord: Coord1D, img: "Image", axis_kind: AxisKind):
    length_val = img.height if axis_kind == AxisKind.Y else img.width
    abs_length = Quantity.points(length_val)

    return coord.to_via_points(UnitType.POINT, abs_length=abs_length)


def mut_coord_to_abs_image(coord: Coord, img: "Image"):
    coord.x = coord1d_to_abs_image(coord.x, img, AxisKind.X)
    coord.y = coord1d_to_abs_image(coord.y, img, AxisKind.Y)
    return coord


@dataclass
class GraphicsObjectConfig:
    children: List["GraphicsObject"] = field(default_factory=list)
    style: Optional[Style] = None
    rotate_in_view: Optional[tuple[float, Point]] = None
    rotate: Optional[float] = None


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

    @property
    def go_type(self) -> GOType:
        raise GGException("not implemented")

    def to_global_coords(self, img: Image):
        raise GGException("Not implented")

    def get_pos(self):
        raise GGException("graphics object has no position")

    def update_view_scale(self, view: "ViewPort"):
        raise GGException("this should never reach")

    def embed_into(self, view: "ViewPort") -> "GraphicsObject":
        from python_ggplot.embed import (
            graphics_object_embed_into,
        )  # pylint: disable=all

        return graphics_object_embed_into(self, view)

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

    def to_global_coords(self, img: Image):
        self.start = mut_coord_to_abs_image(self.start, img)
        self.stop = mut_coord_to_abs_image(self.stop, img)


@dataclass
class GOAxis(GraphicsObject):
    data: StartStopData

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.start)
        view.update_scale(self.data.stop)

    @property
    def go_type(self) -> GOType:
        return GOType.AXIS


@dataclass
class GOLine(GraphicsObject):
    data: StartStopData

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.start)
        view.update_scale(self.data.stop)

    @property
    def go_type(self) -> GOType:
        return GOType.LINE


@dataclass
class TextData:
    text: str
    font: Font
    pos: Coord
    align: TextAlignKind

    def to_global_coords(self, img: Image):
        self.pos = mut_coord_to_abs_image(self.pos, img)


@dataclass
class GOText(GraphicsObject):
    data: TextData

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def get_pos(self):
        return self.data.pos

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.pos)

    @property
    def go_type(self) -> GOType:
        return GOType.TEXT


@dataclass
class GOLabel(GraphicsObject):
    data: TextData

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def get_pos(self):
        return self.data.pos

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.pos)

    @property
    def go_type(self) -> GOType:
        return GOType.LABEL


@dataclass
class GOTickLabel(GraphicsObject):
    data: TextData

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def get_pos(self):
        return self.data.pos

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.pos)

    @property
    def go_type(self) -> GOType:
        return GOType.TICK_LABEL


@dataclass
class GORect(GraphicsObject):
    origin: Coord
    width: Quantity
    height: Quantity

    def to_global_coords(self, img: Image):
        self.origin = mut_coord_to_abs_image(self.origin, img)
        self.width = self.width.to_points(length=Quantity.points(float(img.width)))
        self.height = self.height.to_points(length=Quantity.points(float(img.height)))

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.origin)

    @property
    def go_type(self) -> GOType:
        return GOType.RECT_DATA


@dataclass
class GOGrid(GraphicsObject):
    x_pos: List[Coord1D]
    y_pos: List[Coord1D]
    origin: Optional[Coord] = None
    origin_diagonal: Optional[Coord] = None

    def to_global_coords(self, img: Image):
        if self.origin is None:
            raise GGException("expected origin")

        if self.origin_diagonal is None:
            raise GGException("expected origin_diagonal")

        self.origin = mut_coord_to_abs_image(self.origin, img)
        self.origin_diagonal = mut_coord_to_abs_image(self.origin_diagonal, img)

        img_width = Quantity.points(float(img.width))
        img_height = Quantity.points(float(img.height))

        self.y_pos = [
            item.to_via_points(UnitType.POINT, abs_length=img_height)
            for item in self.y_pos
        ]
        self.x_pos = [
            item.to_via_points(UnitType.POINT, abs_length=img_width)
            for item in self.x_pos
        ]

    def update_view_scale(self, view: "ViewPort"):
        for x_pos in self.x_pos:
            view.update_scale_1d(x_pos)
        for y_pos in self.y_pos:
            view.update_scale_1d(y_pos)

    @property
    def go_type(self) -> GOType:
        return GOType.GRID_DATA


@dataclass
class GOTick(GraphicsObject):
    major: bool
    pos: Coord
    axis: AxisKind
    kind: TickKind
    secondary: bool

    def to_global_coords(self, img: Image):
        self.pos = mut_coord_to_abs_image(self.pos, img)

    def _x_axis_start_stop(self, length: float) -> Tuple[Point, Point]:
        x = self.pos.point().x
        if self.kind == TickKind.ONE_SIDE:
            start = Point(x=x, y=self.pos.point().y + length)
            end = Point(x=x, y=self.pos.point().y)
            return start, end

        elif self.kind == TickKind.BOTH_SIDES:
            start_ = self.pos.point().y + length
            end_ = self.pos.point().y - length
            start = Point(x=x, y=self.pos.point().y + length)
            end = Point(x=x, y=self.pos.point().y - length)
            return start, end
        else:
            raise GGException("unexpected type")

    def _y_axis_start_stop(self, length: float) -> Tuple[Point, Point]:
        y = self.pos.point().y
        if self.kind == TickKind.ONE_SIDE:
            start = Point(x=self.pos.point().x, y=y)
            end = Point(x=self.pos.point().x - length, y=y)
            return start, end

        elif self.kind == TickKind.BOTH_SIDES:
            start_ = self.pos.point().y + length
            end_ = self.pos.point().y - length
            start = Point(x=self.pos.point().x + length, y=y)
            end = Point(x=self.pos.point().x - length, y=y)
            return start, end
        else:
            raise GGException("unexpected type")

    def get_start_stop_point(self, length: float) -> Tuple[Point, Point]:
        if self.axis == AxisKind.X:
            return self._x_axis_start_stop(length)

        elif self.axis == AxisKind.Y:
            return self._y_axis_start_stop(length)
        else:
            raise GGException("unexpected")

    def get_pos(self):
        return self.pos

    def scale_for_axis(self, axis: AxisKind) -> Scale:
        if axis == AxisKind.X:
            return self.pos.x.get_scale()
        if axis == AxisKind.Y:
            return self.pos.y.get_scale()
        raise GGException("unexpected")

    @property
    def go_type(self) -> GOType:
        return GOType.TICK_DATA


@dataclass
class GOPoint(GraphicsObject):
    marker: MarkerKind
    pos: Coord
    size: float
    color: Color

    def to_global_coords(self, img: Image):
        self.pos = mut_coord_to_abs_image(self.pos, img)

    def get_pos(self):
        return self.pos

    @property
    def go_type(self) -> GOType:
        return GOType.POINT_DATA


@dataclass
class GOManyPoints(GraphicsObject):
    marker: MarkerKind
    pos: List[Coord]
    size: float
    color: Color

    def to_global_coords(self, img: Image):
        self.pos = [mut_coord_to_abs_image(pos, img) for pos in self.pos]

    def update_view_scale(self, view: "ViewPort"):
        for pos in self.pos:
            view.update_scale(pos)

    @property
    def go_type(self) -> GOType:
        return GOType.MANY_POINTS_DATA


@dataclass
class GOPolyLine(GraphicsObject):
    pos: List[Coord]

    def to_global_coords(self, img: Image):
        self.pos = [mut_coord_to_abs_image(pos, img) for pos in self.pos]

    def update_view_scale(self, view: "ViewPort"):
        for pos in self.pos:
            view.update_scale(pos)

    @property
    def go_type(self) -> GOType:
        return GOType.POLYLINE_DATA


@dataclass
class GORaster(GraphicsObject):
    origin: Coord
    pixel_width: Quantity
    pixel_height: Quantity
    block_x: int
    block_y: int
    draw_cb: Callable[[], List[int]]

    def to_global_coords(self, img: Image):
        self.origin = mut_coord_to_abs_image(self.origin, img)
        self.pixel_width = self.pixel_width.to_points(
            scale=None, length=Quantity.points(float(img.width))
        )
        self.pixel_height = self.pixel_height.to_points(
            scale=None, length=Quantity.points(float(img.height))
        )

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.origin)

    @property
    def go_type(self) -> GOType:
        return GOType.RASTER_DATA


@dataclass
class GOComposite(GraphicsObject):
    kind: CompositeKind

    def to_global_coords(self, img: Image):
        # nothing to do in this case
        pass

    def update_view_scale(self, view: "ViewPort"):
        if self.config.children is None:
            return

        for go in self.config.children:
            go.update_view_scale(view)

    @property
    def go_type(self) -> GOType:
        return GOType.COMPOSITE_DATA


T = TypeVar("T")


def first_option(left: Optional[T], right: Optional[T]) -> Optional[T]:
    if left is not None:
        return left
    return right


def format_tick_value(f: float, scale: float = 0.0) -> str:
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
