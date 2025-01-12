from copy import deepcopy
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
import typing as tp
from typing import List


from python_ggplot.core_objects import AxisKind, Scale, GGException, Font
from python_ggplot.units import UnitKind, InchUnit, PointUnit, CentimeterUnit
from python_ggplot.cairo_backend import CairoBackend


if tp.TYPE_CHECKING:
    from python_ggplot.units import Quantity


def path_coord_quantity(coord: "Coord1D", length: "Quantity"):
    if coord.is_length_coord:
        length = coord.coord_type.get_length()
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
    left_point = left.to_point()
    right_point = right.to_point()

    length = left_point.get_length()
    pos = operator(left_point.pos, right_point.pos)
    data = LengthCoord(length=length)
    return Coord1D(pos=pos, coord_type=PointCoordType(data=data))


def add_coord_one_length(
    length_coord: "Coord1D", other_coord: "Coord1D", operator: Operator
) -> "Coord1D":
    # todo fix this
    from python_ggplot.units import Quantity

    scale: Scale = other_coord.get_scale()
    length: Quantity = length_coord.get_length()

    left = Quantity(val=length_coord.pos, unit=length_coord.kind)
    right = Quantity(val=other_coord.pos, unit=other_coord.kind)

    quantity = left.apply_operator(right, operator, length, scale, True)

    if quantity.unit.str_type == "relative":
        return Coord1D(pos=quantity.val, coord_type=RelativeCoordType())
    else:
        return Coord1D(pos=quantity.val, coord_type=deepcopy(length_coord.coord_type))


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
        if lhs.is_absolute() and rhs.is_absolute():
            return add_two_absolute_coord(lhs, rhs, operator)
        else:
            res = lhs
            res.pos = operator(lhs.pos, lhs.pos)  # Modify `pos` using the operator
            return res
    elif lhs.is_length_type():
        return add_coord_one_length(lhs, rhs, operator)
    elif rhs.is_length_type():
        return add_coord_one_length(rhs, lhs, operator)
    else:
        left = lhs.to_relative(None)
        right = rhs.to_relative(None)
        pos = operator(left.pos, right.pos)
        return Coord1D(pos=pos, coord_type=RelativeCoordType())


def coord_quantity_operator(
    coord: "Coord1D", quantity: "Quantity", operator: Operator
) -> "Coord1D":
    if coord.str_type != quantity.str_type:
        raise GGException("Quantity and coord types have to be the same")

    pos = operator(coord.pos, quantity.val)

    res = deepcopy(coord)
    res.pos = pos
    return res


def coord_quantity_add(coord: "Coord1D", quantity: "Quantity") -> "Coord1D":
    operator: Operator = lambda a, b: a + b
    coord_quantity_operator(coord, quantity, operator)


def coord_quantity_sub(coord: "Coord1D", quantity: "Quantity") -> "Coord1D":
    operator: Operator = lambda a, b: a - b
    coord_quantity_operator(coord, quantity, operator)


def coord_quantity_mul(coord: "Coord1D", quantity: "Quantity") -> "Coord1D":
    operator: Operator = lambda a, b: a * b
    coord_quantity_operator(coord, quantity, operator)


def coord_quantity_div(coord: "Coord1D", quantity: "Quantity") -> "Coord1D":
    operator: Operator = lambda a, b: a / b
    coord_quantity_operator(coord, quantity, operator)


@dataclass
class Coord1D:
    pos: float
    coord_type: "CoordType"

    @staticmethod
    def from_view(
        unit_kind: "UnitKind", view: "ViewPort", axis_kind: "AxisKind", at: float
    ) -> "Coord1D":
        return unit_kind.from_view(view, axis_kind, at)

    @staticmethod
    def create(view: "ViewPort", at: float, axis_kind: "AxisKind", kind: "UnitKind"):
        coord_type = kind.create_default_coord_type(view, at, axis_kind, kind)
        return Coord1D(pos=at, coord_type=coord_type)

    def to_relative(self, length: Optional["Quantity"]) -> Coord1D:
        return self.coord_type.to_relative(CoordTypeConversion(self, length=length))

    def to_points(
        self,
        length: Optional["Quantity"] = None,
        backend: Optional["CairoBackend"] = None,
    ) -> Coord1D:
        return self.coord_type.to_points(length=length, backend=backend)

    def equal_kind_and_scale(self, other: Coord1D):
        if type(self.coord_type) is not type(other.coord_type):
            return False

        return self.coord_type.compare_scale_and_kind(other)

    def is_absolute(self) -> bool:
        return self.coord_type.is_absolute

    def compatible_kind_and_scale(self, other: "Coord1D"):
        if self.is_absolute() and other.is_absolute():
            return True
        elif self.coord_type != other.coord_type:
            return False
        else:
            return self.equal_kind_and_scale(other)

    def __add__(self, other: "Coord1D") -> "Coord1D":
        if self.compatible_kind_and_scale(other):
            if self.is_absolute() and other.is_absolute():
                left = self.to_points()
                right = other.to_points()
                length_coord = left.data or LengthCoord(length=None)
                return Coord1D(
                    pos=left.pos + right.pos, coord_type=PointCoordType(length_coord)
                )
            else:
                # TODO unit test this
                result = deepcopy(self)
                result.pos = self.pos + other.pos
                return result
        else:
            # TODO fix this.
            from python_ggplot.units import Quantity, RelativeUnit

            if self.coord_type.is_length_coord:
                scale = other.coord_type.data.scale
                added = Quantity(self.pos, self.coord_type).add(
                    Quantity(other.pos, other.coord_type),
                    length=self.coord_type.get_length(),
                    scale=scale,
                    as_coordinate=True,
                )

                # todo fix this
                if isinstance(added, RelativeUnit):
                    return Coord1D(pos=added.val, coord_type=RelativeCoordType())
                else:
                    result = deepcopy(added)
                    result.pos = added.val
                    return result
            else:
                return Coord1D(
                    pos=left.to_relative().pos + other.to_relative().pos,
                    coord_type=RelativeCoordType(),
                )


@dataclass
class CoordTypeConversion:
    coord1d: Coord1D
    length: Optional["Quantity"] = None
    backend: Optional["CairoBackend"] = None


@dataclass
class CoordType:
    str_type = None
    is_absolute = False
    is_length_coord = False

    def from_length(self, length: LengthCoord):
        # todo improve
        raise GGException("This should never be used")

    def get_length(self):
        # todo fix this, fine for now
        return None

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        raise GGException("This should never be used")

    def compare_scale_and_kind(self, other):
        raise GGException("This should never be used")


@dataclass
class LengthCoord:
    length: Optional["Quantity"] = None

    def to_relative_coord1d(
        self, pos: float, length: Optional["Quantity"], kind: UnitKind
    ) -> "Coord1D":
        length = self.length or length
        if length is None:
            raise ValueError("A length is required for relative conversion.")

        relative_length = length.to(kind)
        return Coord1D(pos / relative_length.val, RelativeCoordType())


@dataclass
class DataCoord(CoordType):
    str_type = "data"
    is_absolute = False
    scale: Scale
    axis_kind: AxisKind

    def compare_scale_and_kind(self, other):
        return self.scale == other.scale and self.axis_kind == other.axis_kind

    def to_relative_coord1d(self, pos: float) -> Coord1D:
        if self.axis_kind == AxisKind.X:
            pos = (pos - self.scale.low) / (self.scale.high - self.scale.low)
        elif self.axis_kind == AxisKind.Y:
            pos = 1.0 - (pos - self.scale.low) / (self.scale.high - self.scale.low)
        else:
            raise ValueError("Invalid axis kind")

        return Coord1D(pos=pos, coord_type=RelativeCoordType())


@dataclass
class TextCoord(CoordType):
    str_type = "text"
    is_absolute = False
    text: str
    font: Font

    def to_relative_coord1d(
        self, pos: float, coord_type: CoordType, length: Optional["Quantity"]
    ) -> Coord1D:
        # Get text dimensions
        text_extend = CairoBackend.get_text_extend(self.text, self.font)

        # this has to be str height or str width
        # todo add validation
        dimension = coord_type.text_extend_dimension(text_extend)

        if length is None:
            raise GGException(
                "Conversion from StrWidth to relative requires a length scale!"
            )

        pos = (pos * dimension) / length.to_points(None).val
        return Coord1D(pos=pos, coord_type=RelativeCoordType())


@dataclass
class RelativeCoordType(CoordType):
    str_type = "relative"
    is_length_coord = False

    def compare_scale_and_kind(self, other):
        return True

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return Coord1D(pos=data.coord1d.pos, coord_type=RelativeCoordType())


@dataclass
class PointCoordType(CoordType):
    str_type = "point"
    is_length_coord = True
    is_absolute = True
    data: LengthCoord

    def from_length(self, length: LengthCoord):
        # todo improve
        return PointCoordType(data=length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos, data.length, PointUnit)

    def compare_scale_and_kind(self, other):
        return self.data.length == other.data.length


@dataclass
class CentimeterCoordType(CoordType):
    str_type = "centimeter"
    is_length_coord = True
    is_absolute = True
    data: LengthCoord

    def from_length(self, length: LengthCoord):
        # todo improve
        return PointCoordType(data=length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(
            data.coord1d.pos, data.length, CentimeterUnit
        )

    def compare(self, other):
        return self.data.length == other.data.length


@dataclass
class InchCoordType(CoordType):
    str_type = "inch"
    is_absolute = True
    is_length_coord = True
    data: LengthCoord

    def from_length(self, length: LengthCoord):
        # todo improve
        return PointCoordType(data=length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos, data.length, InchUnit)

    def compare(self, other):
        return self.data.length == other.data.length


@dataclass
class DataCoordType(CoordType):
    str_type = "data"
    data: DataCoord

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos)


@dataclass
class StrWidthCoordType(CoordType):
    str_type = "str_width"
    is_absolute = True
    data: TextCoord

    def text_extend_dimension(self, text_extend):
        return text_extend.width()

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos, self, data.length)


@dataclass
class StrHeightCoordType(CoordType):
    str_type = "str_height"
    is_absolute = True
    data: TextCoord

    def text_extend_dimension(self, text_extend):
        return text_extend.height()

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        raise GGException("Not implemented")


@dataclass
class Coord:
    x: Coord1D
    y: Coord1D

    def to_relative(self) -> "Coord":
        x = self.x.to_relative(None)
        y = self.y.to_relative(None)
        return Coord(x=x, y=y)

    def __eq__(self, other: "Coord") -> bool:
        return self.x == other.x and self.y == other.y


@dataclass
class GridCoord:
    origin: Coord
    origin_diagonal: Coord
    x: List[Coord]
    y: List[Coord]
