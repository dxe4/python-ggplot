import typing as tp
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from python_ggplot.cairo_backend import CairoBackend
from python_ggplot.common import abs_to_inch, inch_to_abs, inch_to_cm
from python_ggplot.core_objects import AxisKind, Font, GGException, Scale
from python_ggplot.units import (CentimeterUnit, InchUnit, PointUnit,
                                 RelativeUnit, UnitKind)


def quantitiy_to_coord(quantity):
    conversion_data = {
        "relative": lambda: RelativeCoordType(quantity.pos),
        "point": lambda: PointCoordType(quantity.pos, data=LengthCoord(length=deepcopy(quantity))),
        "inch": lambda: InchCoordType(quantity.pos, data=LengthCoord(length=deepcopy(quantity))),
        "centimeter": lambda: CentimeterCoordType(
            quantity.pos, data=LengthCoord(length=deepcopy(quantity))
        ),
    }
    conversion = conversion_data[quantity.unit.str_type]
    return conversion()



def unit_to_point(str_type, pos):
    data = {
        "centimeter": lambda: inch_to_abs(abs_to_inch(pos)),
        "point": lambda: pos,
        "inch": lambda: inch_to_abs(pos),
    }
    convert = data.get(str_type)
    if not convert:
        raise GGException("convert not possible")
    return convert()


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
    return PointCoordType(pos=pos, data=data)


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
        return RelativeCoordType(pos=quantity.val)
    else:
        result = deepcopy(length_coord.coord_type)
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
        if lhs.is_absolute and rhs.is_absolute:
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
        return RelativeCoordType(pos=pos)


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
class ToCord:
    to_kind: UnitKind = RelativeUnit()
    length: Optional["Quantity"] = None
    abs_length: Optional["Quantity"] = None
    scale: Optional[Scale] = None
    axis: Optional[AxisKind] = None
    str_text: Optional[str] = None
    str_font: Optional[str] = None

def default_coord_view_location(view: 'ViewPort', kind: AxisKind):
    if kind == AxisKind.X:
        return view.point_width(), view.x_scale
    elif kind == AxisKind.Y:
        return view.point_height(), view.y_scale
    else:
        raise GGException("")

@dataclass
class Coord1D:
    pos: float
    str_type = None
    is_absolute = False
    is_length_coord = False

    def default_length_and_scale(self, view: 'ViewPort', kind: "UnitKind"):
        length, scale = default_coord_view_location(view, kind)
        return length, scale


    @staticmethod
    def create_default_coord_type(
        view: 'ViewPort', at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> "CoordType":
        raise GGException("not implemented")

    @staticmethod
    def from_view(
        view: "ViewPort", axis_kind: "AxisKind", at: float
    ) -> "Coord1D":
        raise GGException("Not implemented")

    def embed_into(self, axis_kind: AxisKind, into: 'ViewPort'):
        if self.is_length_coord:
            origin, abs_length = into.embed_into_origin_for_length(axis_kind)
            data = ToCord(abs_length=abs_length)
            origin_abs = origin.to_point(data)
            return origin_abs + self
        else:
            origin, abs_length = into.embed_into_origin()
            pos = (origin.pos * abs_length.val) * self.to_relative(ToCord()).pos
            return RelativeCoordType(pos)

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        raise GGException("not implemented")

    def from_length(self, length: LengthCoord):
        raise GGException("This should never be used")

    def get_length(self):
        # todo fix this, fine for now
        return None


    def to_inch(self, convert_data: ToCord):
        raise GGException("This should never be sued")

    def to_centimeter(self, convert_data: ToCord):
        raise GGException("This should never be sued")

    def to_point(self, convert_data: ToCord):
        raise GGException("This should never be sued")

    def to_relative(self, data: ToCord) -> 'Coord1D':
        raise GGException("This should never be used")

    def compare_scale_and_kind(self, other):
        raise GGException("This should never be used")

    def to(self, to_kind: UnitKind, data: ToCord):
        # todo refactor function
        from python_ggplot.units import ToQuantityData  # todo fix
        if self.str_type == to_kind.str_type:
            return deepcopy(self)

        if self.is_length_coord:
            conversaion_table = {
                "centimeter": self.to_centimeter,
                "point": self.to_point,
                "inch": self.to_inch,
            }
            convert = conversaion_table.get(to_kind.str_type)
            if not convert:
                raise GGException("conversion not possible")
            return convert(data)

        rel = self.to_relative(data)
        if to_kind.str_type == "relative":
            return rel

        if to_kind.str_type == "data":
            if data.axis is None:
                raise GGException("expected axis")
            if data.scale is None:
                raise GGException("expected scale")
            pos = (data.scale.high - data.scale.low) * rel.pos + data.scale.low
            return DataCoordType(pos=pos, data=DataCoord(scale=data.scale, axis_kind=data.axis))

        if to_kind.str_type in ("str_width", "str_height"):
            raise GGException("cannot convert")

        if to_kind.str_type == "point":
            if data.abs_length is None:
                raise GGException("expected abs length")
            length = data.abs_length.to_points(ToQuantityData())
            pos = rel.pos * length.val
            return PointUnit(pos=pos, data=LengthCoord(length=length))

        if to_kind.str_type == "centimeter":
            if data.abs_length is None:
                raise GGException("expected abs length")
            length = data.abs_length.to_centimeter(ToQuantityData())
            pos = rel.pos * length.val
            return CentimeterCoordType(pos=pos, data=LengthCoord(length=length))

        if to_kind.str_type == "inch":
            if data.abs_length is None:
                raise GGException("expected abs length")
            length = data.abs_length.to_inch(ToQuantityData())
            pos = rel.pos * length.val
            return InchCoordType(pos=pos, data=LengthCoord(length=length))

        raise GGException("shouldnt reach here")

    @staticmethod
    def create(view: "ViewPort", at: float, axis_kind: "AxisKind", kind: "UnitKind"):
        return kind.create_default_coord_type(view, at, axis_kind, kind)

    def equal_kind_and_scale(self, other: 'Coord1D'):
        if self.str_type != other.str_type:
            return False

        return self.compare_scale_and_kind(other)

    def compatible_kind_and_scale(self, other: "Coord1D"):
        if self.is_absolute and other.is_absolute:
            return True
        elif self.str_type != other.str_type:
            return False
        else:
            return self.equal_kind_and_scale(other)

    def __add__(self, other: "Coord1D") -> "Coord1D":
        if self.compatible_kind_and_scale(other):
            if self.is_absolute and other.is_absolute:
                left = self.to_point(ToCord())
                right = other.to_points()
                length_coord = left.data or LengthCoord(length=None)
                return PointCoordType(pos=left.pos + right.pos, data=length_coord)
            else:
                # TODO unit test this
                result = deepcopy(self)
                result.pos = self.pos + other.pos
                return result
        else:
            # TODO fix this.
            from python_ggplot.units import Quantity, RelativeUnit

            if self.is_length_coord:
                scale = other.data.scale
                added = Quantity(self.pos, self.coord_type).add(
                    Quantity(other.pos, other.coord_type),
                    length=self.coord_type.get_length(),
                    scale=scale,
                    as_coordinate=True,
                )

                # todo fix this
                if isinstance(added, RelativeUnit):
                    return RelativeCoordType(pos=added.val)
                else:
                    result = deepcopy(added)
                    result.pos = added.val
                    return result
            else:
                return RelativeCoordType(pos=left.to_relative().pos + other.to_relative().pos)


@dataclass
class LengthCoord:
    length: Optional["Quantity"] = None


@dataclass
class DataCoord():
    str_type = "data"
    is_absolute = False
    scale: Scale
    axis_kind: AxisKind

    def compare_scale_and_kind(self, other):
        return self.scale == other.scale and self.axis_kind == other.axis_kind

@dataclass
class TextCoordData:
    is_absolute = False
    text: str
    font: Font

    def get_text_extend(self):
        return CairoBackend.get_text_extend(self.text, self.font)


@dataclass
class RelativeCoordType(Coord1D):
    str_type = "relative"
    is_length_coord = False

    @staticmethod
    def create_default_coord_type(
        view: 'ViewPort', at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> 'Coord1D':
        return RelativeCoordType(pos=at)

    def compare_scale_and_kind(self, other):
        return True

    def to_relative(self, data: ToCord) -> Coord1D:
        return RelativeCoordType(pos=data.coord1d.pos)

    def to_point(self, data: ToCord) -> Coord1D:
        if not data.length:
            raise GGException("expected length for conversion")
        return PointCoordType(pos=self.pos + data.length.val, data=LengthCoord(length=data.length))


class LengthCoordMixin:
    is_length_coord = True
    is_absolute = True

    def _to_point(self, data: ToCord):
        from python_ggplot.units import convert_quantity_data  # todo fix this
        res_length = None
        if data.length:
            res_length = convert_quantity_data(PointUnit, data.length, None)
        pos = unit_to_point(self.str_type, self.pos)
        return PointCoordType(pos=pos, data=LengthCoord(length=res_length))

    def _to_relative(
        self, data: LengthCoord, pos: float, length: Optional["Quantity"], kind: UnitKind
    ) -> "Coord1D":
        length = data.length or length
        if length is None:
            raise ValueError("A length is required for relative conversion.")

        relative_length = length.to(kind)
        return RelativeCoordType(pos / relative_length.val)


    def _to_centimeter(self) -> Coord1D:
        # point_pos = coord1d.to_points().pos
        pos = inch_to_cm(abs_to_inch(self.pos))
        length = self.data.length.to_centimeter()
        # todo make helper func
        return CentimeterCoordType(pos=pos, data=LengthCoord(length=length))

    def _to_inch(self) -> Coord1D:
        # point_pos = coord1d.to_points().pos
        pos = abs_to_inch(self.pos)
        length = self.data.length.to_inch()
        # todo make helper func
        return InchCoordType(pos=pos, data=LengthCoord(length=length))


@dataclass
class PointCoordType(Coord1D, LengthCoordMixin):
    str_type = "point"
    data: LengthCoord

    @staticmethod
    def create_default_coord_type(
        view: 'ViewPort', at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> "CoordType":
        length, _ = super().default_length_and_scale(view, kind)
        return PointCoordType(pos=at, data=LengthCoord(length=length.to_points()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float) -> "Coord1D":
        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_points(length)
        return PointCoordType(pos=at, data=LengthCoord(length=length))

    def to_point(self, convert_data: ToCord):
        return self._to_point(self.data)

    def to_inch(self, convert_data: ToCord):
        return self._to_inch()

    def to_centimeter(self, convert_data: ToCord) -> Coord1D:
        return self._to_centimeter()

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        self.data.length = view.length_from_axis(axis_kind)

    def from_length(self, length: LengthCoord):
        return PointCoordType(pos=self.pos, data=length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def to_relative(self, data: ToCord) -> Coord1D:
        return self._to_relative(self.data, data.coord1d.pos, data.length, PointUnit)

    def compare_scale_and_kind(self, other):
        return self.data.length == other.data.length


@dataclass
class CentimeterCoordType(Coord1D, LengthCoordMixin):
    str_type = "centimeter"
    is_length_coord = True
    is_absolute = True
    data: LengthCoord

    def create_default_coord_type(
        self, view: ViewPort, at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> 'Coord1D':
        length, _ = super().default_length_and_scale(view, kind)
        return CentimeterCoordType(pos=at, data=LengthCoord(length=length.to_centimeter()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float
    ) -> "Coord1D":
        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_points(length)
        return CentimeterCoordType(pos=at, data=LengthCoord(length=length))


    def to_point(self, convert_data: ToCord):
        return self._to_point(convert_data)

    def to_centimeter(self, convert_data: ToCord) -> Coord1D:
        return deepcopy(self)

    def to_inch(self, convert_data: ToCord):
        return self._to_inch()

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        self.data.length = view.length_from_axis(axis_kind)

    def from_length(self, length: LengthCoord):
        return PointCoordType(pos=self.pos, data=length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def to_relative(self, data: ToCord) -> Coord1D:
        return self._to_relative(
            self.data, data.coord1d.pos, data.length, CentimeterUnit
        )

    def compare(self, other):
        return self.data.length == other.data.length


@dataclass
class InchCoordType(Coord1D, LengthCoordMixin):
    str_type = "inch"
    is_absolute = True
    is_length_coord = True
    data: LengthCoord

    @staticmethod
    def create_default_coord_type(
        view: 'ViewPort', at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> 'Coord1D':
        length, _ = super().default_length_and_scale(view, kind)
        return InchCoordType(pos=at, data=LengthCoord(length=length.to_inch()))

    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float
    ) -> "Coord1D":

        length: Quantity = view.length_from_axis(axis_kind)
        length = length.to_points(length)
        return InchCoordType(pos=at, data=LengthCoord(length=length))

    def to_point(self, convert_data: ToCord):
        return self._to_point(convert_data)

    def to_inch(self):
        return deepcopy(self)

    def to_centimeter(self, convert_data: ToCord) -> Coord1D:
        return self._to_centimeter()

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        self.data.length = view.length_from_axis(axis_kind)

    def from_length(self, length: LengthCoord):
        return PointCoordType(pos=self.pos, data=length)

    def get_length(self):
        # todo fix this, fine for now
        return self.data.length

    def to_relative(self, data: ToCord) -> Coord1D:
        return self._to_relative(
            self.data, data.coord1d.pos, data.length, InchUnit
        )

    def compare(self, other):
        return self.data.length == other.data.length


@dataclass
class DataCoordType(Coord1D):
    str_type = "data"
    data: DataCoord

    @staticmethod
    def create_default_coord_type(
        view: 'ViewPort', at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> 'Coord1D':
        _, scale = super().default_length_and_scale(view, kind)
        data = DataCoord(scale=scale, axis_kind=axis_kind)
        return DataCoordType(pos=at, data=data)


    @staticmethod
    def from_view(view: "ViewPort", axis_kind: "AxisKind", at: float
    ) -> "Coord1D":
        scale = view.scale_for_axis(axis_kind)
        return DataCoordType(pos=at, data=DataCoord(scale=scale, axis_kind=axis_kind))

    def to(self, to_kind: UnitKind, data: ToCord):
        result = super().to(to_kind, data)
        if result:
            return result

        conversaion_table = {
            "point": self.to_point,
        }
        convert = conversaion_table.get(to_kind.str_type)
        if not convert:
            raise GGException("conversion not possible")
        return convert(data)

    def to_point(self, convert_data: ToCord):
        return self.to_relative(convert_data).to_point(convert_data.length)

    def _to_relative_x(self):
        return (self.pos - self.data.scale.low) / (self.data.scale.high - self.data.scale.low)

    def _to_relative_y(self):
        return 1.0 - (self.pos - self.data.scale.low) / (self.data.scale.high - self.data.scale.low)

    def _to_relative_data(self):
        if self.data.axis_kind == AxisKind.X:
            return self._to_relative_x()
        if self.data.axis_kind == AxisKind.Y:
            return self._to_relative_y()
        raise GGException()

    def to_relative(self, convert_data: ToCord):
        return RelativeCoordType(pos=self._to_relative_data())

    def update_from_view(self, view: "ViewPort", axis_kind: AxisKind):
        scale = view.scale_for_axis(axis_kind)
        self.data.scale = scale

class TextCoordTypeMixin:

    def to_point(self, convert_data: ToCord):
        if not convert_data.length:
            raise GGException("length must be provided")

        dimension = self.point_dimension()
        pos = self.pos * dimension
        return PointCoordType(pos=pos, data=LengthCoord(length=convert_data.length))

    def _to_relative(self, length: Optional["Quantity"]) -> Coord1D:
        if length is None:
            raise GGException(
                "Conversion from StrWidth to relative requires a length scale!"
            )

        text_extend = CairoBackend.get_text_extend(self.text, self.font)
        # this has to be str height or str width
        # todo add validation
        dimension = self.text_extend_dimension(text_extend)
        pos = (self.pos * dimension) / length.to_points(None).val
        return RelativeCoordType(pos=pos)


@dataclass
class StrWidthCoordType(Coord1D, TextCoordTypeMixin):
    str_type = "str_width"
    is_absolute = True
    data: TextCoordData

    def point_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.x_bearing() + text_extend.x_advance()

    def relative_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.width()

    def text_extend_dimension(self, text_extend):
        return text_extend.width()

    def to_relative(self, data: ToCord) -> Coord1D:
        return self._to_relative(data.length)

    def to_point(self, data: ToCord) -> Coord1D:
        return self._to_relative(data.length)


@dataclass
class StrHeightCoordType(Coord1D, TextCoordTypeMixin):
    str_type = "str_height"
    is_absolute = True
    data: TextCoordData

    def relative_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.height()

    def point_dimension(self):
        text_extend = self.data.get_text_extend()
        return text_extend.y_bearing() + text_extend.y_advance()

    def to_point(self, data: ToCord) -> Coord1D:
        return self._to_relative(data.length)

    def text_extend_dimension(self, text_extend):
        return text_extend.height()

    def to_relative(self, data: ToCord) -> Coord1D:
        return self._to_relative(data.length)


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
