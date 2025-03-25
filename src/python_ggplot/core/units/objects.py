from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Type

from python_ggplot.core.objects import AxisKind, GGException, Scale, UnitType

if TYPE_CHECKING:
    from python_ggplot.core.coord.objects import OperatorType
    from python_ggplot.graphics.views import ViewPort


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
class Quantity(ABC):
    val: float

    @staticmethod
    def from_type(unit_type: UnitType, val: float) -> "Quantity":
        return unit_type_from_type(unit_type)(val)

    @staticmethod
    def from_type_or_none(
        unit_type: UnitType, val: Optional[float]
    ) -> Optional["Quantity"]:
        if not val:
            return None
        return unit_type_from_type(unit_type)(val)

    @property
    @abstractmethod
    def unit_type(self) -> UnitType:
        pass

    @staticmethod
    def centimeters(val: float) -> "Quantity":
        return CentimeterUnit(val=val)

    @staticmethod
    def points(val: float) -> "Quantity":
        return PointUnit(val=val)

    @staticmethod
    def inches(val: float) -> "Quantity":
        return InchUnit(val=val)

    @staticmethod
    def relative(val: float) -> "Quantity":
        return RelativeUnit(val=val)

    def embed_into(self, axis: AxisKind, view: "ViewPort") -> "Quantity":
        from python_ggplot.core.embed import quantity_embed_into  # pylint: disable=all

        return quantity_embed_into(self, axis, view)

    def to_relative_with_view(self, view: "ViewPort", axis: AxisKind):
        length = view.to_relative_dimension(axis)
        return self.to_relative(length=length, scale=deepcopy(view.x_scale))

    def to_relative_from_view(
        self, view: "ViewPort", axis_kind: AxisKind
    ) -> "Quantity":
        from python_ggplot.core.units.convert import (
            to_relative_from_view,
        )  # pylint: disable=all

        return to_relative_from_view(self, view, axis_kind)

    def to(
        self,
        kind: UnitType,
        length: Optional["Quantity"] = None,
        scale: Optional[Scale] = None,
    ) -> "Quantity":
        from python_ggplot.core.units.convert import (
            convert_quantity,
        )  # pylint: disable=all

        return convert_quantity(self, kind, length=length, scale=scale)

    def to_data(
        self, length: Optional["Quantity"] = None, scale: Optional[Scale] = None
    ) -> "Quantity":
        return self.to(UnitType.DATA, length=length, scale=scale)

    def to_centimeter(
        self, length: Optional["Quantity"] = None, scale: Optional[Scale] = None
    ) -> "Quantity":
        return self.to(UnitType.CENTIMETER, length=length, scale=scale)

    def to_inch(
        self, length: Optional["Quantity"] = None, scale: Optional[Scale] = None
    ) -> "Quantity":
        return self.to(UnitType.INCH, length=length, scale=scale)

    def to_points(
        self, length: Optional["Quantity"] = None, scale: Optional[Scale] = None
    ) -> "Quantity":
        return self.to(UnitType.POINT, length=length, scale=scale)

    def to_relative(
        self, length: Optional["Quantity"] = None, scale: Optional[Scale] = None
    ) -> "Quantity":
        return self.to(UnitType.RELATIVE, length=length, scale=scale)

    def apply_operator(
        self,
        other: "Quantity",
        length: Optional["Quantity"],
        scale: Optional[Scale],
        as_coordinate: bool,  # noqa TODO fix
        operator: Callable[[float, float], float],
        operator_type: "OperatorType",
    ) -> "Quantity":
        # TODO fix circular import
        from python_ggplot.core.coord.objects import OperatorType

        # todo refactor
        # this is ugly, needs to become like the conversion eventually
        if self.unit_type == other.unit_type:
            cls = unit_type_from_type(self.unit_type)
            return cls(operator(self.val, other.val))
        elif self.unit_type.is_length() and other.unit_type == UnitType.RELATIVE:
            if operator_type in {OperatorType.MUL, OperatorType.DIV}:
                return PointUnit(operator(self.val, other.val)).to(self.unit_type)
            else:
                return PointUnit(
                    operator(self.to_points().val, other.to_points(length=length).val)
                ).to(self.unit_type, length=length, scale=scale)
        elif self.unit_type.is_length() and (other.unit_type.is_length()):
            return PointUnit(
                operator(self.to_points().val, other.to_points(length=length).val)
            ).to(self.unit_type, length=length, scale=scale)
        elif self.unit_type == UnitType.RELATIVE and other.unit_type.is_length():
            if operator_type in {OperatorType.MUL, OperatorType.DIV}:
                return PointUnit(operator(self.val, other.val)).to(other.unit_type)
            else:
                return PointUnit(
                    operator(self.to_points(length=length).val, other.to_points().val)
                ).to(self.unit_type, length=length, scale=scale)
        elif self.unit_type == UnitType.DATA:
            left = deepcopy(self)
            if as_coordinate:
                if not scale:
                    raise GGException("Scale is needed to convert unity type DATA")
                left = DataUnit(self.val - scale.low)
            left = left.to_relative(length=length, scale=scale).val
            right = other.to_relative(length=length, scale=scale).val
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
        # TODO fix circular import
        from python_ggplot.core.coord.objects import OperatorType

        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a * b, OperatorType.MUL
        )

    def add(
        self,
        other: "Quantity",
        length: Optional["Quantity"] = None,
        scale: Optional[Scale] = None,
        as_coordinate: bool = False,
    ) -> "Quantity":
        # TODO fix circular import
        from python_ggplot.core.coord.objects import OperatorType

        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a + b, OperatorType.ADD
        )

    def divide(
        self,
        other: "Quantity",
        length: Optional["Quantity"] = None,
        scale: Optional[Scale] = None,
        as_coordinate: bool = False,
    ) -> "Quantity":
        # TODO fix circular import
        from python_ggplot.core.coord.objects import OperatorType

        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a / b, OperatorType.DIV
        )

    def subtract(
        self,
        other: "Quantity",
        length: Optional["Quantity"] = None,
        scale: Optional[Scale] = None,
        as_coordinate: bool = False,
    ) -> "Quantity":
        # TODO fix circular import
        from python_ggplot.core.coord.objects import OperatorType

        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a - b, OperatorType.SUB
        )


class PointUnit(Quantity):

    @property
    def unit_type(self) -> UnitType:
        return UnitType.POINT


class CentimeterUnit(Quantity):

    @property
    def unit_type(self) -> UnitType:
        return UnitType.CENTIMETER


class InchUnit(Quantity):

    @property
    def unit_type(self) -> UnitType:
        return UnitType.INCH


class RelativeUnit(Quantity):

    @property
    def unit_type(self) -> UnitType:
        return UnitType.RELATIVE


class DataUnit(Quantity):

    @property
    def unit_type(self) -> UnitType:
        return UnitType.DATA


class StrWidthUnit(Quantity):

    @property
    def unit_type(self) -> UnitType:
        return UnitType.STR_WIDTH


class StrHeightUnit(Quantity):

    @property
    def unit_type(self) -> UnitType:
        return UnitType.STR_HEIGHT


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


def unit_type_from_type(kind: UnitType) -> Type[Quantity]:
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
