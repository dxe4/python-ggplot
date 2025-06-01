from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from python_ggplot.common.maths import create_curve
from python_ggplot.core.common import REPR_CONFIG
from python_ggplot.core.coord.objects import Coord, Coord1D, DataCoord, DataCoordType
from python_ggplot.core.objects import (
    BLACK,
    TRANSPARENT,
    AxisKind,
    Color,
    CompositeKind,
    Font,
    GGEnum,
    GGException,
    Image,
    LineType,
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
    from python_ggplot.graphics.views import ViewPort


def coord1d_to_abs_image(coord: Coord1D, img: "Image", axis_kind: AxisKind):
    if axis_kind == AxisKind.Y:
        length_val = img.height
    elif axis_kind == AxisKind.X:
        length_val = img.width
    else:
        raise GGException("expect x or y axis")

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
    rotate_in_view: Optional[tuple[float, Point[float]]] = None
    rotate: Optional[float] = None

    def __rich_repr__(self):
        if REPR_CONFIG["GO_RECURSIVE"]:
            yield "children", self.children
        if REPR_CONFIG["GO_STYLE"]:
            yield "style", self.style
        yield "rotate", self.rotate
        yield "rotate_in_view", self.rotate_in_view


class GOType(GGEnum):
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
class GraphicsObject(ABC):
    name: str
    config: GraphicsObjectConfig

    @abstractmethod
    def get_coords(self) -> Dict[Any, Any]:
        pass

    @property
    @abstractmethod
    def go_type(self) -> GOType:
        pass

    @abstractmethod
    def to_global_coords(self, img: Image):
        pass

    @abstractmethod
    def get_pos(self) -> "Coord":
        pass

    @abstractmethod
    def update_view_scale(self, view: "ViewPort"):
        pass

    def embed_into(self, view: "ViewPort") -> "GraphicsObject":
        from python_ggplot.core.embed import (
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

    def get_coords(self) -> Dict[Any, Any]:
        return {
            "start": self.start,
            "end": self.stop,
        }

    def to_global_coords(self, img: Image):
        self.start = mut_coord_to_abs_image(self.start, img)
        self.stop = mut_coord_to_abs_image(self.stop, img)


@dataclass
class GOAxis(GraphicsObject):
    data: StartStopData

    def get_coords(self) -> Dict[Any, Any]:
        data = {"name": self.name, "type": self.__class__.__name__}
        data.update(self.data.get_coords())
        return data

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.start)
        view.update_scale(self.data.stop)

    @property
    def go_type(self) -> GOType:
        return GOType.AXIS

    def get_pos(self) -> "Coord":
        raise GGException("not implemented")


@dataclass
class GOLine(GraphicsObject):
    data: StartStopData

    def get_coords(self) -> Dict[Any, Any]:
        data = {"name": self.name, "type": self.__class__.__name__}
        data.update(self.data.get_coords())
        return data

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.start)
        view.update_scale(self.data.stop)

    @property
    def go_type(self) -> GOType:
        return GOType.LINE

    def get_pos(self) -> "Coord":
        raise GGException("not implemented")


@dataclass
class TextData:
    text: str
    font: Font
    pos: Coord
    align: TextAlignKind

    def get_coords(self) -> Dict[Any, Any]:
        return {"pos": self.pos}

    def to_global_coords(self, img: Image):
        self.pos = mut_coord_to_abs_image(self.pos, img)


@dataclass
class GOText(GraphicsObject):
    data: TextData

    def get_coords(self) -> Dict[Any, Any]:
        data = {"name": self.name, "type": self.__class__.__name__}
        data.update(self.data.get_coords())
        return data

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def get_pos(self) -> "Coord":
        return self.data.pos

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.pos)

    @property
    def go_type(self) -> GOType:
        return GOType.TEXT


@dataclass
class GOLabel(GraphicsObject):
    data: TextData

    def get_coords(self) -> Dict[Any, Any]:
        data = {"name": self.name, "type": self.__class__.__name__}
        data.update(self.data.get_coords())
        return data

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def get_pos(self) -> "Coord":
        return self.data.pos

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.data.pos)

    @property
    def go_type(self) -> GOType:
        return GOType.LABEL


@dataclass
class GOTickLabel(GraphicsObject):
    data: TextData

    def get_coords(self) -> Dict[Any, Any]:
        data = {"name": self.name, "type": self.__class__.__name__}
        data.update(self.data.get_coords())
        return data

    def to_global_coords(self, img: Image):
        self.data.to_global_coords(img)

    def get_pos(self) -> "Coord":
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

    def get_coords(self) -> Dict[Any, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "origin": self.origin,
            "width": self.width,
            "height": self.height,
        }

    def to_global_coords(self, img: Image):
        self.origin = mut_coord_to_abs_image(self.origin, img)
        self.width = self.width.to_points(length=Quantity.points(float(img.width)))
        self.height = self.height.to_points(length=Quantity.points(float(img.height)))

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.origin)

    def get_pos(self) -> "Coord":
        raise GGException("not implemented")

    @property
    def go_type(self) -> GOType:
        return GOType.RECT_DATA


@dataclass
class GOGrid(GraphicsObject):
    x_pos: List[Coord1D]
    y_pos: List[Coord1D]
    origin: Optional[Coord] = None
    origin_diagonal: Optional[Coord] = None

    # TODO double check this, original package seems to start with
    # Relative 0.0 for both origin and origin diagonal
    # embed_into sets the right origin later, so it doesnt seem to matter
    # need to check if this has any impact
    def get_coords(self) -> Dict[Any, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "x_pos": self.x_pos,
            "y_pos": self.y_pos,
            "origin": self.origin,
            "origin_diagonal": self.origin_diagonal,
        }

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

    def get_pos(self) -> "Coord":
        raise GGException("not implemented")

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

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.pos)

    def get_coords(self) -> Dict[Any, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "pos": self.pos,
        }

    def to_global_coords(self, img: Image):
        self.pos = mut_coord_to_abs_image(self.pos, img)

    def _x_axis_start_stop(self, length: float) -> Tuple[Point[float], Point[float]]:
        x = self.pos.point().x
        if self.kind == TickKind.ONE_SIDE:
            start = Point(x=x, y=self.pos.point().y + length)
            end = Point(x=x, y=self.pos.point().y)
            return start, end

        elif self.kind == TickKind.BOTH_SIDES:
            start = Point(x=x, y=self.pos.point().y + length)
            end = Point(x=x, y=self.pos.point().y - length)
            return start, end
        else:
            raise GGException("unexpected type")

    def _y_axis_start_stop(self, length: float) -> Tuple[Point[float], Point[float]]:
        y = self.pos.point().y
        if self.kind == TickKind.ONE_SIDE:
            start = Point(x=self.pos.point().x, y=y)
            end = Point(x=self.pos.point().x - length, y=y)
            return start, end

        elif self.kind == TickKind.BOTH_SIDES:
            start = Point(x=self.pos.point().x + length, y=y)
            end = Point(x=self.pos.point().x - length, y=y)
            return start, end
        else:
            raise GGException("unexpected type")

    def get_start_stop_point(self, length: float) -> Tuple[Point[float], Point[float]]:
        if self.axis == AxisKind.X:
            return self._x_axis_start_stop(length)

        elif self.axis == AxisKind.Y:
            return self._y_axis_start_stop(length)
        else:
            raise GGException("unexpected")

    def get_pos(self) -> "Coord":
        return self.pos

    def scale_for_axis(self, axis: AxisKind) -> Scale:
        # TODO low priority, easy fix
        # fix the type here, its not critical and it will work fine
        if axis == AxisKind.X:
            return self.pos.x.get_scale()  # type: ignore
        if axis == AxisKind.Y:
            return self.pos.y.get_scale()  # type: ignore
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

    def get_coords(self) -> Dict[Any, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "pos": self.pos,
        }

    def to_global_coords(self, img: Image):
        self.pos = mut_coord_to_abs_image(self.pos, img)

    def get_pos(self) -> "Coord":
        return self.pos

    @property
    def go_type(self) -> GOType:
        return GOType.POINT_DATA

    def update_view_scale(self, view: "ViewPort"):
        view.update_scale(self.pos)


@dataclass
class GOManyPoints(GraphicsObject):
    marker: MarkerKind
    pos: List[Coord]
    size: float
    color: Color

    def get_coords(self) -> Dict[Any, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "pos": self.pos,
        }

    def to_global_coords(self, img: Image):
        self.pos = [mut_coord_to_abs_image(pos, img) for pos in self.pos]

    def update_view_scale(self, view: "ViewPort"):
        for pos in self.pos:
            view.update_scale(pos)

    @property
    def go_type(self) -> GOType:
        return GOType.MANY_POINTS_DATA

    def get_pos(self) -> "Coord":
        raise GGException("not implemented")


@dataclass
class GOPolyLine(GraphicsObject):
    pos: List[Coord]

    def get_coords(self) -> Dict[Any, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "pos": self.pos,
        }

    def to_global_coords(self, img: Image):
        self.pos = [mut_coord_to_abs_image(pos, img) for pos in self.pos]

    def update_view_scale(self, view: "ViewPort"):
        for pos in self.pos:
            view.update_scale(pos)

    @property
    def go_type(self) -> GOType:
        return GOType.POLYLINE_DATA

    def get_pos(self) -> "Coord":
        raise GGException("not implemented")


class GOCurve(GOPolyLine):

    def create(
        self,
        x: Union[float, int],
        y: Union[float, int],
        xend: Union[float, int],
        yend: Union[float, int],
        curvature: Union[float, int],
        x_scale: Scale,
        y_scale: Scale,
        name: Optional[str] = None,
        style: Optional[Style] = None,
    ):

        curve_points = create_curve(x, y, xend, yend, curvature)
        curve_positions = [
            Coord(
                x=DataCoordType(
                    pos=curve_point.x,
                    data=DataCoord(
                        axis_kind=AxisKind.X,
                        scale=x_scale,
                    ),
                ),
                y=DataCoordType(
                    pos=curve_point.y,
                    data=DataCoord(
                        axis_kind=AxisKind.Y,
                        scale=y_scale,
                    ),
                ),
            )
            for curve_point in curve_points
        ]
        style = style or Style(
            line_width=2.0,
            line_type=LineType.SOLID,
            color=BLACK,
            fill_color=TRANSPARENT,
        )

        return GOPolyLine(
            name=name or "curve",
            config=GraphicsObjectConfig(style=style),
            pos=curve_positions,
        )


@dataclass
class GORaster(GraphicsObject):
    origin: Coord
    pixel_width: Quantity
    pixel_height: Quantity
    block_x: int
    block_y: int
    draw_cb: Callable[[], List[int]]

    def get_coords(self) -> Dict[Any, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "origin": self.origin,
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
        }

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

    def get_pos(self) -> "Coord":
        raise GGException("not implemented")


@dataclass
class GOComposite(GraphicsObject):
    kind: CompositeKind

    def get_coords(self) -> Dict[Any, Any]:
        return {}

    def get_pos(self) -> "Coord":
        raise GGException("not implemented")

    def to_global_coords(self, img: Image):
        # nothing to do in this case
        pass

    def update_view_scale(self, view: "ViewPort"):
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
