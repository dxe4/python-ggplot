from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Protocol, Type, cast

from python_ggplot.core.common import LOG_LEVEL, abs_to_inch, inch_to_abs, inch_to_cm
from python_ggplot.core.objects import (
    AxisKind,
    Font,
    GGException,
    Point,
    Scale,
    UnitType,
)
from python_ggplot.core.units.objects import Quantity, unit_type_from_type
from python_ggplot.graphics.cairo_backend import CairoBackend

if TYPE_CHECKING:
    from python_ggplot.graphics.views import ViewPort


def coord_type_from_unit_type(unit_type: UnitType):
    lookup = {
        UnitType.POINT: PointCoordType,
        UnitType.CENTIMETER: CentimeterCoordType,
        UnitType.INCH: InchCoordType,
        UnitType.RELATIVE: RelativeCoordType,
        UnitType.DATA: DataCoordType,
        UnitType.CENTIMETER: InchCoordType,
        UnitType.STR_HEIGHT: StrHeightCoordType,
        UnitType.STR_WIDTH: StrWidthCoordType,
    }
    return lookup[unit_type]


def default_coord_view_location(view: "ViewPort", kind: AxisKind):
    length_data = {
        AxisKind.X: (view.point_width(), view.x_scale),
        AxisKind.Y: (view.point_height(), view.y_scale),
    }
    return length_data[kind]


def default_length_and_scale(view: "ViewPort", kind: AxisKind):
    length, scale = default_coord_view_location(view, kind)
    return length, scale


def path_coord_quantity(coord: "Coord1D", length: Quantity):
    if isinstance(coord, (CentimeterCoordType, InchCoordType, PointCoordType)):
        length = coord.get_length()
        if length is None:
            coord.data.length = length
    return coord


def path_coord_view_port(coord: "Coord", view: "ViewPort") -> "Coord":
    return Coord(
        x=path_coord_quantity(coord.x, view.w_img),
        y=path_coord_quantity(coord.y, view.h_img),
    )


Operator = Callable[[float, float], float]


def add_two_absolute_coord(
    left: "Coord1D", right: "Coord1D", operator: Operator
) -> "Coord1D":
    left_point = left.to(UnitType.POINT)
    right_point = right.to(UnitType.POINT)

    length = left_point.get_length()
    pos = operator(left_point.pos, right_point.pos)
    data = LengthCoord(length=length)
    return PointCoordType(pos, data)


def add_coord_one_length(
    length_coord: "Coord1D", other_coord: "Coord1D", operator: Operator
) -> "Coord1D":
    scale: Scale = other_coord.get_scale()
    length: Quantity = length_coord.get_length()

    left_cls = unit_type_from_type(length_coord.unit_type)
    right_cls = unit_type_from_type(other_coord.unit_type)
    left = left_cls(val=length_coord.pos)
    right = right_cls(val=other_coord.pos)

    quantity = left.apply_operator(right, operator, length, scale, True)

    if quantity.unit_type == UnitType.RELATIVE:
        return RelativeCoordType(pos=quantity.val)
    else:
        result = deepcopy(length_coord)
        result.pos = quantity.val
        return result


def coord_operator(
    lhs: "Coord1D", rhs: "Coord1D", operator: Operator, operator_type: "OperatorType"
) -> "Coord1D":
    """
    # TODO this allows you to pass operator=lambda x,y: x-y and OperatorType.DIV
    needs refactoring to become prone to errors
    """
    # todo unit tests
    alike = False
    if operator_type == OperatorType.DIV:
        alike = lhs.equal_kind_and_scale(rhs)
    else:
        alike = lhs.compatible_kind_and_scale(rhs)

    if alike:
        if lhs.unit_type.is_absolute() and rhs.unit_type.is_absolute():
            return add_two_absolute_coord(lhs, rhs, operator)
        else:
            res = deepcopy(lhs)
            res.pos = operator(lhs.pos, rhs.pos)
            return res
    elif lhs.unit_type.is_length():
        return add_coord_one_length(lhs, rhs, operator)
    elif rhs.unit_type.is_length():
        return add_coord_one_length(rhs, lhs, operator)
    else:
        left = lhs.to(UnitType.RELATIVE)
        right = rhs.to(UnitType.RELATIVE)
        pos = operator(left.pos, right.pos)
        return RelativeCoordType(pos=pos)


def coord_quantity_operator(
    coord: "Coord1D", quantity: Quantity, operator: Operator
) -> "Coord1D":
    if coord.unit_type != quantity.unit_type:
        raise GGException("Quantity and coord types have to be the same")

    res = deepcopy(coord)
    res.pos = operator(coord.pos, quantity.val)
    return res


def coord_quantity_add(coord: "Coord1D", quantity: Quantity) -> "Coord1D":
    operator: Operator = lambda a, b: a + b
    return coord_quantity_operator(coord, quantity, operator)


def coord_quantity_sub(coord: "Coord1D", quantity: Quantity) -> "Coord1D":
    operator: Operator = lambda a, b: a - b
    return coord_quantity_operator(coord, quantity, operator)


def coord_quantity_mul(coord: "Coord1D", quantity: Quantity) -> "Coord1D":
    operator: Operator = lambda a, b: a * b
    return coord_quantity_operator(coord, quantity, operator)


def coord_quantity_div(coord: "Coord1D", quantity: Quantity) -> "Coord1D":
    operator: Operator = lambda a, b: a / b
    return coord_quantity_operator(coord, quantity, operator)


@dataclass
class CoordsInput:
    left: float = 0.0
    bottom: float = 0.0
    width: float = 1.0
    height: float = 1.0


class OperatorType(Enum):
    DIV = "DIV"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"


@dataclass
class ToCord:
    cord: "Coord1D"
    to_kind: UnitType = UnitType.RELATIVE
    length: Optional[Quantity] = None
    abs_length: Optional[Quantity] = None
    scale: Optional[Scale] = None
    axis: Optional[AxisKind] = None
    str_text: Optional[str] = None
    str_font: Optional[str] = None


@dataclass
class Coord1D:
    pos: float

    @property
    def unit_type(self) -> UnitType:
        raise GGException("not implemented")

    @staticmethod
    def create_str_height(pos: float, font: Font) -> "StrHeightCoordType":
        return StrHeightCoordType(pos, data=TextCoordData(text="W", font=font))

    @staticmethod
    def create_str_width(pos: float, font: Font) -> "StrWidthCoordType":
        return StrWidthCoordType(pos, data=TextCoordData(text="W", font=font))

    @staticmethod
    def create_relative(pos: float) -> "Coord1D":
        return RelativeCoordType(pos)

    @staticmethod
    def create_data(pos: float, scale: Scale, axis_kind: AxisKind) -> "Coord1D":
        return DataCoordType(pos, data=DataCoord(scale=scale, axis_kind=axis_kind))

    @staticmethod
    def create_point(pos: float, length: Optional[Quantity] = None) -> "Coord1D":
        return PointCoordType(pos, data=LengthCoord(length=length))

    def update_scale(self, view: "ViewPort"):
        # only applicable to DataCoord
        pass

    def __eq__(self, other) -> bool:
        if self.unit_type.is_length() and other.unit_type.is_length():
            return self.to_points().pos == other.to_points().pos
        else:
            return self.to_relative().pos == other.to_relative().pos

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        cls = coord_type_from_unit_type(kind)
        return cls.create_default_coord_type(view, at, axis_kind, kind)

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        raise GGException("Not implemented")

    def embed_into(self, axis_kind: AxisKind, into: "ViewPort") -> "Coord1D":
        from python_ggplot.embed import coord1d_embed_into

        return coord1d_embed_into(self, axis_kind, into)

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        raise GGException("not implemented")

    def from_length(self, length: "LengthCoord"):
        raise GGException("This should never be used")

    def get_length(self):
        # todo fix this, fine for now
        return None

    def get_scale(self):
        # todo fix this, fine for now
        return None

    def to_inches(self, length=None):
        return self.to(UnitType.INCH, length=length)

    def to_centimeters(self, length=None):
        return self.to(UnitType.CENTIMETER, length=length)

    def to_points(self, length=None):
        return self.to(UnitType.POINT, length=length)

    def to_relative(self, length=None) -> "Coord1D":
        return self.to(UnitType.RELATIVE, length=length)

    def to_via_points(
        self,
        to_kind: UnitType,
        length: Optional[Quantity] = None,
        abs_length: Optional[Quantity] = None,
        scale: Optional[Scale] = None,
        axis: Optional[AxisKind] = None,
    ) -> "Coord1D":
        from python_ggplot.core.coord.convert import (
            convert_via_point,
        )  # pylint: disable=all

        return convert_via_point(
            self, to_kind, length=length, abs_length=abs_length, scale=scale, axis=axis
        )

    def to(self, to_kind: UnitType, length: Optional[Quantity] = None) -> "Coord1D":
        from python_ggplot.core.coord.convert import (
            convert_coord,
        )  # pylint: disable=all

        return convert_coord(coord=self, to_type=to_kind, length=length)

    @staticmethod
    def create(view: "ViewPort", at: float, axis_kind: "AxisKind", kind: Type):
        return kind.create_default_coord_type(view, at, axis_kind, kind)

    def equal_kind_and_scale(self, other: "Coord1D"):
        if self.unit_type != other.unit_type:
            return False

        if self.unit_type.is_length():
            return self == other

        if self.unit_type == UnitType.DATA and other.unit_type == UnitType.DATA:
            self_ = cast(DataCoordType, self)
            other_ = cast(DataCoordType, other)
            return (
                self_.data.scale == other_.data.scale
                and self_.data.axis_kind == other_.data.axis_kind
            )

        if self.unit_type == UnitType.RELATIVE:
            return True

        raise GGException("This should never be reached")

    def compatible_kind_and_scale(self, other: "Coord1D"):
        if self.unit_type.is_absolute() and other.unit_type.is_absolute():
            return True
        elif self.unit_type != other.unit_type:
            return False
        else:
            return self.equal_kind_and_scale(other)

    def __add__(self, other: "Coord1D") -> "Coord1D":
        return coord_operator(self, other, lambda x, y: x + y, OperatorType.ADD)

    def __mul__(self, other: "Coord1D") -> "Coord1D":
        return coord_operator(self, other, lambda x, y: x * y, OperatorType.MUL)

    def __truediv__(self, other: "Coord1D") -> "Coord1D":
        return coord_operator(self, other, lambda x, y: x / y, OperatorType.DIV)

    def __sub__(self, other: "Coord1D") -> "Coord1D":
        return coord_operator(self, other, lambda x, y: x - y, OperatorType.SUB)


@dataclass
class LengthCoord:
    length: Optional[Quantity] = None


@dataclass
class DataCoord:
    axis_kind: AxisKind
    scale: Scale


@dataclass
class TextCoordData:
    text: str
    font: Font

    def get_text_extend(self):
        return CairoBackend.get_text_extend(self.text, self.font)


@dataclass
class RelativeCoordType(Coord1D):

    @property
    def unit_type(self) -> UnitType:
        return UnitType.RELATIVE

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        return RelativeCoordType(pos=at)


@dataclass
class PointCoordType(Coord1D):
    data: LengthCoord = field(default_factory=LengthCoord)

    @property
    def unit_type(self) -> UnitType:
        return UnitType.POINT

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        length, _ = default_length_and_scale(view, axis_kind)
        return PointCoordType(at, LengthCoord(length=length.to_points()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_points()
        return PointCoordType(at, LengthCoord(length=length))

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        self.data.length = view.length_from_axis(axis_kind)

    def from_length(self, length: LengthCoord):
        return PointCoordType(self.pos, length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length


@dataclass
class CentimeterCoordType(Coord1D):
    data: LengthCoord

    @property
    def unit_type(self) -> UnitType:
        return UnitType.CENTIMETER

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        length, _ = default_length_and_scale(view, axis_kind)
        return CentimeterCoordType(at, LengthCoord(length=length.to_centimeter()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        # nom source -> template c1*
        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_centimeter()
        return CentimeterCoordType(at, LengthCoord(length=length))

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        self.data.length = view.length_from_axis(axis_kind)

    def from_length(self, length: LengthCoord):
        return PointCoordType(self.pos, length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def compare(self, other):
        return self.data.length == other.data.length


@dataclass
class InchCoordType(Coord1D):
    data: LengthCoord

    @property
    def unit_type(self) -> UnitType:
        return UnitType.INCH

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        length, _ = default_length_and_scale(view, axis_kind)
        return InchCoordType(at, LengthCoord(length=length.to_inch()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        # nom source -> template c1*
        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_inch()
        return InchCoordType(at, LengthCoord(length=length))

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        self.data.length = view.length_from_axis(axis_kind)

    def from_length(self, length: LengthCoord):
        return PointCoordType(self.pos, length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def compare(self, other):
        return self.data.length == other.data.length


@dataclass
class DataCoordType(Coord1D):
    data: DataCoord

    @property
    def unit_type(self) -> UnitType:
        return UnitType.DATA

    def update_scale(self, view: "ViewPort"):
        if self.data.axis_kind == AxisKind.X:
            self.scale = view.x_scale
        if self.data.axis_kind == AxisKind.Y:
            self.scale = view.y_scale

    def get_scale(self):
        # todo fix this, fine for now
        return self.data.scale

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        _, scale = default_length_and_scale(view, axis_kind)
        data = DataCoord(scale=scale, axis_kind=axis_kind)
        return DataCoordType(at, data)

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        # template c1*
        scale = view.scale_for_axis(axis_kind)
        if scale is None:
            raise GGException("expected scale")
        return DataCoordType(at, DataCoord(scale=scale, axis_kind=axis_kind))

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        scale = view.scale_for_axis(axis_kind)
        if scale is None:
            raise GGException("expected scale")
        self.data.scale = scale


@dataclass
class StrWidthCoordType(Coord1D):
    data: TextCoordData

    @property
    def unit_type(self) -> UnitType:
        return UnitType.STR_WIDTH

    def point_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.x_bearing() + text_extend.x_advance()

    def relative_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.width()

    def text_extend_dimension(self, text_extend):
        return text_extend.width()


@dataclass
class StrHeightCoordType(Coord1D):
    data: TextCoordData

    @property
    def unit_type(self) -> UnitType:
        return UnitType.STR_HEIGHT

    def relative_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.height

    def point_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.y_bearing() + text_extend.y_advance()

    def text_extend_dimension(self, text_extend):
        return text_extend.height


@dataclass
class Coord:
    x: Coord1D
    y: Coord1D

    def point(self) -> Point[float]:
        return Point(
            x=self.x.pos,
            y=self.y.pos,
        )

    def dimension_for_axis(self, axis: AxisKind) -> "Coord1D":
        if axis == AxisKind.X:
            return self.x
        else:
            return self.y

    @staticmethod
    def relative(x: float, y: float) -> "Coord":
        return Coord(
            x=RelativeCoordType(x),
            y=RelativeCoordType(y),
        )

    def to_relative(self) -> "Coord":
        x = self.x.to(UnitType.RELATIVE)
        y = self.y.to(UnitType.RELATIVE)
        return Coord(x=x, y=y)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def embed_into(self, into: "ViewPort") -> "Coord":
        from python_ggplot.embed import coord_embed_into

        return coord_embed_into(self, into)


@dataclass
class GridCoord:
    origin: Coord
    origin_diagonal: Coord
    x: List[Coord]
    y: List[Coord]


def coord_type_from_type(kind: UnitType):
    data = {
        UnitType.POINT: PointCoordType,
        UnitType.CENTIMETER: CentimeterCoordType,
        UnitType.INCH: InchCoordType,
        UnitType.RELATIVE: RelativeCoordType,
        UnitType.DATA: DataCoordType,
        UnitType.STR_WIDTH: StrWidthCoordType,
        UnitType.STR_HEIGHT: StrWidthCoordType,
    }
    return data[kind]
