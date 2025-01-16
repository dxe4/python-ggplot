from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from python_ggplot.common import abs_to_inch, cm_to_inch, inch_to_abs, inch_to_cm
from python_ggplot.core_objects import AxisKind, GGException, Scale, UnitType

if TYPE_CHECKING:
    from python_ggplot.views import ViewPort


def unity_type_to_quantity_cls(kind: UnitType):
    data = {
        UnitType.CENTIMETER: CentimeterUnit,
        UnitType.POINT: PointUnit,
        UnitType.INCH: InchUnit,
        UnitType.DATA: DataUnit,
        UnitType.RELATIVE: RelativeUnit,
        UnitType.STR_WIDTH: StrWidthUnit,
        UnitType.STR_HEIGHT: StrHeightUnit,
    }
    return data[kind]


@dataclass
class Quantity:
    val: float
    unit_type: UnitType
    str_type = None
    is_length_unit: bool = False

    @staticmethod
    def centimeters(val: float) -> "Quantity":
        return CentimeterUnit(val)

    @staticmethod
    def points(val: float) -> "Quantity":
        return PointUnit(val)

    @staticmethod
    def relative(val: float) -> "Quantity":
        return RelativeUnit(val)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embed_into(self, axis: AxisKind, view: "ViewPort") -> "Quantity":
        from python_ggplot.embed import quantity_embed_into  # pylint: disable=all

        return quantity_embed_into(self, axis, view)

    def to_relative_with_view(self, view: "ViewPort", axis: AxisKind):
        length = view.to_relative_dimension(axis)
        return self.to_relative(length=length, scale=deepcopy(view.x_scale))

    def to_relative_from_view(
        self, view: "ViewPort", axis_kind: AxisKind
    ) -> "Quantity":
        from python_ggplot.quantity_convert import (
            to_relative_from_view,
        )  # pylint: disable=all

        return to_relative_from_view(self, view, axis_kind)

    def to(self, kind: UnitType, length=None, scale=None):
        from python_ggplot.quantity_convert import (
            convert_quantity,
        )  # pylint: disable=all

        return convert_quantity(self, kind, length=length, scale=scale)

    def to_data(self, length=None, scale=None):
        return self.to(UnitType.DATA, length=length, scale=scale)

    def to_centimeter(self, length=None, scale=None):
        return self.to(UnitType.CENTIMETER, length=length, scale=scale)

    def to_inch(self, length=None, scale=None):
        return self.to(UnitType.INCH, length=length, scale=scale)

    def to_points(self, length=None, scale=None):
        return self.to(UnitType.POINT, length=length, scale=scale)

    def to_relative(self, length=None, scale=None):
        return self.to(UnitType.RELATIVE, length=length, scale=scale)

    def apply_operator(
        self,
        other: "Quantity",
        length: Optional["Quantity"],
        scale: Optional[Scale],
        as_coordinate: bool,  # noqa TODO fix
        operator: Callable[[float, float], float],
    ) -> "Quantity":
        # todo refactor
        # this is ugly, needs to become like the conversion eventually
        if self.unit_type == other.unit_type:
            cls = unit_type_from_type(self.unit_type)
            return cls(operator(self.val, other.val))
        elif self.unit_type.is_length():
            other_converted = other.to(UnitType.POINT, length=length)
            result_val = operator(self.to(UnitType.POINT).val, other_converted.val)
            return PointUnit(result_val).to(self.unit_type, length, scale)
        elif self.unit_type == RelativeUnit and self.unit_type == RelativeUnit:
            raise ValueError("Cannot perform arithmetic on two RELATIVE quantities")
        elif self.unit_type == RelativeUnit:
            other_converted = other.to(UnitType.POINT, length=length)
            cls = unity_type_to_quantity_cls(self.unit_type)
            return cls(operator(self.val, other_converted.val))
        elif isinstance(self.unit_type, DataUnit):
            if not scale or not as_coordinate:
                raise GGException("TODO re evaluate nim version of this")

            left = self.to(UnitType.RELATIVE, length=length, scale=scale).val
            right = other.to(UnitType.RELATIVE, length=length, scale=scale).val
            return RelativeUnit(operator(left, right))
        else:
            raise GGException(f"Unsupported unit arithmetic for {self.unit_type}")

    def multiply(
        self,
        other: "Quantity",
        length: Optional["Quantity"] = None,
        scale: Optional[Scale] = None,
        as_coordinate: bool = False,
    ) -> "Quantity":
        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a * b
        )

    def add(
        self,
        other: "Quantity",
        length: Optional["Quantity"] = None,
        scale: Optional[Scale] = None,
        as_coordinate: bool = False,
    ) -> "Quantity":
        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a + b
        )

    def divide(
        self,
        other: "Quantity",
        length: Optional["Quantity"] = None,
        scale: Optional[Scale] = None,
        as_coordinate: bool = False,
    ) -> "Quantity":
        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a / b
        )

    def subtract(
        self,
        other: "Quantity",
        length: Optional["Quantity"] = None,
        scale: Optional[Scale] = None,
        as_coordinate: bool = False,
    ) -> "Quantity":
        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a - b
        )


class PointUnit(Quantity):

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.POINT
        super().__init__(*args, **kwargs)


class CentimeterUnit(Quantity):

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.CENTIMETER
        super().__init__(*args, **kwargs)


class InchUnit(Quantity):

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.INCH
        super().__init__(*args, **kwargs)


class RelativeUnit(Quantity):

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.RELATIVE
        super().__init__(*args, **kwargs)


class DataUnit(Quantity):

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.DATA
        super().__init__(*args, **kwargs)


class StrWidthUnit(Quantity):

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.STR_WIDTH
        super().__init__(*args, **kwargs)


class StrHeightUnit(Quantity):

    def __init__(self, *args, **kwargs):
        kwargs["unit_type"] = UnitType.STR_HEIGHT
        super().__init__(*args, **kwargs)


def add_data_quantity(
    left: Quantity,
    right: Quantity,
    length: Optional[Quantity],
    scale: Optional[Scale],
    operator: Callable[[float, float], float],
) -> Quantity:
    if scale is None:
        raise ValueError("Expected scale")

    final_q = DataUnit(left.val - scale.low)

    left_relative = final_q.to_relative(length=length, scale=scale)
    right_relative = right.to_relative(length=length, scale=scale)

    val = operator(left_relative.val, right_relative.val)

    return RelativeUnit(val)


def add_length_relative_quantity(
    length_quantity: Quantity,
    relative_quantity: Quantity,
    operator: Callable[[float, float], float],
) -> Quantity:
    val = operator(length_quantity.val, relative_quantity.val)
    cls = unit_type_from_type(length_quantity.unit_type)
    return cls(val)


def add_length_quantities(
    left: Quantity, right: Quantity, operator: Callable[[float, float], float]
) -> Quantity:

    left_converted = left.to(UnitType.POINT)
    right_converted = right.to(UnitType.POINT)

    val = operator(left_converted.val, right_converted.val)
    point = PointUnit(val)
    return point.to(left.unit_type)


def unit_type_from_type(kind: UnitType):
    data = {
        UnitType.POINT: PointUnit,
        UnitType.CENTIMETER: CentimeterUnit,
        UnitType.INCH: InchUnit,
        UnitType.RELATIVE: RelativeUnit,
        UnitType.DATA: DataUnit,
        UnitType.STR_WIDTH: StrWidthUnit,
        UnitType.STR_HEIGHT: StrHeightUnit,
    }
    return data[kind]
