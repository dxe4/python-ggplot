from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Protocol, Type

from python_ggplot.cairo_backend import CairoBackend
from python_ggplot.common import abs_to_inch, inch_to_abs, inch_to_cm
from python_ggplot.core_objects import (AxisKind, Font, GGException, Scale,
                                        UnitType)
from python_ggplot.units import Quantity, unit_type_from_type

if TYPE_CHECKING:
    from python_ggplot.graphics_objects import ViewPort


def path_coord_quantity(coord: "Coord1D", length: Quantity):
    if coord.unit_type.is_length():
        length = coord.get_length()
        if length is None:
            coord.coord_type.length = length
    return coord


def path_coord_view_port(coord: "Coord", view: "ViewPort") -> "Coord":
    return Coord(
        x=path_coord_quantity(coord.x, view.w_img),
        y=path_coord_quantity(coord.y, view.h_img),
    )


Operator = Callable[[float, float], float]


class OperatorType(Enum):
    DIV = "DIV"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"


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
    lhs: "Coord1D", rhs: "Coord1D", operator: Operator, operator_type: OperatorType
) -> "Coord1D":
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
            res = lhs
            res.pos = operator(lhs.pos, lhs.pos)  # Modify `pos` using the operator
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

    pos = operator(coord.pos, quantity.val)

    res = deepcopy(coord)
    res.pos = pos
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
class ToCord:
    cord: "Coord1D"
    to_kind: UnitType = UnitType.RELATIVE
    length: Optional[Quantity] = None
    abs_length: Optional[Quantity] = None
    scale: Optional[Scale] = None
    axis: Optional[AxisKind] = None
    str_text: Optional[str] = None
    str_font: Optional[str] = None


def default_coord_view_location(view: "ViewPort", kind: AxisKind):
    if kind == AxisKind.X:
        return view.point_width(), view.x_scale
    elif kind == AxisKind.Y:
        return view.point_height(), view.y_scale
    else:
        raise GGException("")


def default_length_and_scale(view: "ViewPort", kind: AxisKind):
    length, scale = default_coord_view_location(view, kind)
    return length, scale


@dataclass
class Coord1D:
    pos: float
    unit_type: UnitType

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        raise GGException("not implemented")

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        raise GGException("Not implemented")

    def embed_into(self, axis_kind: AxisKind, into: "ViewPort"):
        if self.unit_type.is_length():
            origin, abs_length = into.coord_embed_into_origin_for_length(axis_kind)
            origin_abs = origin.to_via_points(abs_length=abs_length)
            return origin_abs + self
        else:
            origin, abs_length = into.coord_embed_into_origin(axis_kind)
            pos = (origin.pos * abs_length.val) * self.to_relative().pos
            return RelativeCoordType(pos)

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        raise GGException("not implemented")

    def from_length(self, length: LengthCoord):
        raise GGException("This should never be used")

    def get_length(self):
        # todo fix this, fine for now
        return None

    def get_scale(self):
        # todo fix this, fine for now
        return None

    def to_inch(self, length=None):
        return self.to(UnitType.INCH, length=length)

    def to_centimeter(self, length=None):
        return self.to(UnitType.CENTIMETER, length=length)

    def to_point(self, length=None):
        return self.to(UnitType.POINT, length=length)

    def to_relative(self, length=None) -> "Coord1D":
        return self.to(UnitType.RELATIVE, length=length)

    def compare_scale_and_kind(self, other):
        raise GGException("This should never be used")

    def to_via_points(
        self, to_kind: UnitType, length=None, abs_length=None, scale=None, axis=None
    ):
        from python_ggplot.coord_convert import convert_via_point

        return convert_via_point(
            self, to_kind, length=None, abs_length=None, scale=None, axis=None
        )

    def to(self, to_kind: UnitType, length=None) -> "Coord1D":
        from python_ggplot.coord_convert import convert_coord

        return convert_coord(coord=self, to_type=to_kind, length=length)

    @staticmethod
    def create(view: "ViewPort", at: float, axis_kind: "AxisKind", kind: Type):
        return kind.create_default_coord_type(view, at, axis_kind, kind)

    def equal_kind_and_scale(self, other: "Coord1D"):
        if self.unit_type != other.unit_type:
            return False

        return self.compare_scale_and_kind(other)

    def compatible_kind_and_scale(self, other: "Coord1D"):
        if self.unit_type.is_absolute() and other.unit_type.is_absolute():
            return True
        elif self.unit_type != other.unit_type:
            return False
        else:
            return self.equal_kind_and_scale(other)

    def __add__(self, other: "Coord1D") -> "Coord1D":
        if self.compatible_kind_and_scale(other):
            if self.unit_type.is_absolute() and other.unit_type.is_absolute():
                left = self.to(UnitType.POINT)
                right = other.to(UnitType.POINT)
                length_coord = LengthCoord(length=left.get_length())
                return PointCoordType(left.pos + right.pos, length_coord)
            else:
                # TODO unit test this
                result = deepcopy(self)
                result.pos = self.pos + other.pos
                return result
        else:
            if self.unit_type.is_length():
                scale = other.get_scale()
                left_cls = unit_type_from_type(self.unit_type)
                right_cls = unit_type_from_type(other.unit_type)

                added = left_cls(self.pos).add(
                    right_cls(other.pos),
                    length=self.get_length(),
                    scale=scale,
                    as_coordinate=True,
                )

                if added.unit_type == UnitType.RELATIVE:
                    return RelativeCoordType(pos=added.val)
                else:
                    result = deepcopy(added)
                    result.pos = added.val
                    return result
            else:
                return RelativeCoordType(
                    pos=left.to(UnitType.RELATIVE).pos + other.to(UnitType.RELATIVE).pos
                )


@dataclass
class LengthCoord:
    length: Optional[Quantity] = None


@dataclass
class DataCoord:
    scale: Scale
    axis_kind: AxisKind

    def compare_scale_and_kind(self, other):
        return self.scale == other.scale and self.axis_kind == other.axis_kind


@dataclass
class TextCoordData:
    text: str
    font: Font

    def get_text_extend(self):
        return CairoBackend.get_text_extend(self.text, self.font)


@dataclass
class RelativeCoordType(Coord1D):

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.RELATIVE
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        return RelativeCoordType(pos=at)

    def compare_scale_and_kind(self, other):
        return True


@dataclass
class PointCoordType(Coord1D):
    data: LengthCoord

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.POINT
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        length, _ = default_length_and_scale(view, axis_kind)
        return PointCoordType(at, LengthCoord(length=length.to_points()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_points(length=length)
        return PointCoordType(at, LengthCoord(length=length))

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        self.data.length = view.length_from_axis(axis_kind)

    def from_length(self, length: LengthCoord):
        return PointCoordType(self.pos, length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def compare_scale_and_kind(self, other):
        return self.data.length == other.data.length


@dataclass
class CentimeterCoordType(Coord1D):
    data: LengthCoord

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.CENTIMETER
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        length, _ = default_length_and_scale(view, axis_kind)
        return CentimeterCoordType(at, LengthCoord(length=length.to_centimeter()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_points(length)
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

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.INCH
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        length, _ = default_length_and_scale(view, axis_kind)
        return InchCoordType(at, LengthCoord(length=length.to_inch()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":

        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_points(length)
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

    def get_scale(self):
        # todo fix this, fine for now
        return self.data.scale

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.DATA
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_default_coord_type(
        view: "ViewPort", at: float, axis_kind: AxisKind, kind: UnitType
    ) -> "Coord1D":
        _, scale = default_length_and_scale(view, axis_kind)
        data = DataCoord(scale=scale, axis_kind=axis_kind)
        return DataCoordType(at, data)

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        scale = view.scale_for_axis(axis_kind)
        return DataCoordType(at, DataCoord(scale=scale, axis_kind=axis_kind))

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        scale = view.scale_for_axis(axis_kind)
        self.data.scale = scale


@dataclass
class StrWidthCoordType(Coord1D):
    data: TextCoordData

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.STR_WIDTH
        super().__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.STR_HEIGHT
        super().__init__(*args, **kwargs)

    def relative_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.height()

    def point_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.y_bearing() + text_extend.y_advance()

    def text_extend_dimension(self, text_extend):
        return text_extend.height()


@dataclass
class Coord:
    x: Coord1D
    y: Coord1D

    def to_relative(self) -> "Coord":
        x = self.x.to(UnitType.RELATIVE)
        y = self.y.to(UnitType.RELATIVE)
        return Coord(x=x, y=y)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y


@dataclass
class GridCoord:
    origin: Coord
    origin_diagonal: Coord
    x: List[Coord]
    y: List[Coord]
