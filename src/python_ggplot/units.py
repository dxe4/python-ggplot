from dataclasses import dataclass
from typing import Callable, Optional

from python_ggplot.common import abs_to_inch, cm_to_inch, inch_to_abs, inch_to_cm
from python_ggplot.core_objects import GGException


@dataclass
class ToQuantityData:
    scale: Optional["Scale"] = None
    length: Optional["Quantity"] = None


@dataclass
class QuantityConversionData:
    quantity: "Quantity"
    length: Optional["Quantity"] = None
    scale: Optional["Scale"] = None

    def validate_generic_conversion(self, kind):
        if kind.str_type == self.quantity.unit.str_type:
            return Quantity(val=self.quantity.val, unit=self.quantity.unit)
        if kind.str_type == "data" and not self.scale:
            raise GGException("cannot covnert to data without scale")

    def validate_to_relative_conversion(self):
        if self.quantity.unit.str_type not in ["data", "relative"]:
            if self.length and self.quantity.unit in [
                PointUnit,
                CentimeterUnit,
                InchUnit,
            ]:
                raise GGException(
                    "length scale needed to convert quantity to relative value!"
                )
        elif isinstance(self.quantity.unit, DataUnit) and self.scale:
            raise GGException(
                "length scale needed to convert quantity to relative value!"
            )


@dataclass
class Quantity:
    val: float
    unit: "UnitKind"

    def to(self, kind, length=None, scale=None):
        data = QuantityConversionData(quantity=self, length=length, scale=scale)
        data.validate_generic_conversion(kind)
        conversaion_table = {
            PointUnit: self.to_points,
            CentimeterUnit: self.to_centimeter,
            InchUnit: self.to_inch,
            RelativeUnit: self.to_relative,
            DataUnit: self.to_data,
        }

        return conversaion_table[kind](data)

    def to_data(self, quantity_data: ToQuantityData):
        data = QuantityConversionData(
            quantity=self, length=quantity_data.length, scale=quantity_data.scale
        )
        return self.unit.to_data(self, data)

    def to_centimeter(self, quantity_data: ToQuantityData):
        data = QuantityConversionData(quantity=self)
        return self.unit.to_centimeter(data)

    def to_inch(self, quantity_data: ToQuantityData):
        data = QuantityConversionData(quantity=self)
        return self.unit.to_inch(data)

    def to_points(self, quantity_data: ToQuantityData):
        data = QuantityConversionData(quantity=self, length=quantity_data.length)
        return self.unit.to_points(data)

    def to_relative(self, quantity_data: ToQuantityData):
        data = QuantityConversionData(
            quantity=self, length=quantity_data.length, scale=quantity_data.scale
        )
        data.validate_to_relative_conversion()
        return self.unit.to_relative(data)

    def apply_operator(
        self,
        other: "Quantity",
        length: Optional["Quantity"],
        scale: Optional["Scale"],
        as_coordinate: bool,  # noqa TODO fix
        operator: Callable[[float, float], float],
    ) -> "Quantity":
        # todo refactor
        if self.unit.str_type == other.unit.str_type:
            return Quantity(operator(self.val, other.val), self.unit)
        if self.unit.is_length_unit:
            other_converted = other.to_points(length)
            result_val = operator(
                self.to_points(ToQuantityData()).val, other_converted.val
            )
            return Quantity(result_val, PointUnit).to(self.unit, length, scale)
        elif isinstance(self.unit, RelativeUnit):
            if isinstance(self.unit, RelativeUnit):
                raise ValueError("Cannot perform arithmetic on two RELATIVE quantities")
            other_converted = other.to_points(length)
            return Quantity(operator(self.val, other_converted.val), self.unit)
        elif isinstance(self.unit, DataUnit):
            if not scale:
                raise GGException("TODO re evaluate nim version of this")
            if not as_coordinate:
                raise GGException("TODO re evaluate nim version of this")

            left = self.to_relative(ToQuantityData(length=length, scale=scale)).val
            right = other.to_relative(length, scale).val
            return Quantity(operator(left, right), RelativeUnit)
        else:
            raise GGException(f"Unsupported unit arithmetic for {self.unit}")

    def multiply(
        self,
        other: "Quantity",
        length: Optional["Quantity"],
        scale: Optional["Scale"],
        as_coordinate: bool = False,
    ) -> "Quantity":
        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a * b
        )

    def add(
        self,
        other: "Quantity",
        length: Optional["Quantity"],
        scale: Optional["Scale"],
        as_coordinate: bool = False,
    ) -> "Quantity":
        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a + b
        )

    def divide(
        self,
        other: "Quantity",
        length: Optional["Quantity"],
        scale: Optional["Scale"],
        as_coordinate: bool = False,
    ) -> "Quantity":
        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a / b
        )

    def subtract(
        self,
        other: "Quantity",
        length: Optional["Quantity"],
        scale: Optional["Scale"],
        as_coordinate: bool = False,
    ) -> "Quantity":
        return self.apply_operator(
            other, length, scale, as_coordinate, lambda a, b: a - b
        )


class UnitKind:
    str_type = None
    is_length_unit = False

    def is_point(self):
        # todo temp hack
        return False

    def to_data(self, data: QuantityConversionData):
        return GGException("not implemented")

    def to_centimeter(self, data: QuantityConversionData):
        raise GGException("not implemented")

    def to_inch(self, data: QuantityConversionData):
        raise GGException("not implemented")

    def to_relative(self, data: QuantityConversionData):
        raise GGException("not implemented")

    def to_points(self, data: QuantityConversionData):
        raise GGException("to points not implement for this type")


class PointUnit(UnitKind):
    str_type = "point"
    is_length_unit = True

    def is_point(self):
        # todo temp
        return True

    def to_data(self, data: QuantityConversionData):
        new_val = (data.scale.high - data.scale.low) * data.quantity.to_relative(
            data.length, data.scale
        ).val
        return Quantity(val=new_val, unit=DataUnit())

    def to_centimeter(self, data: QuantityConversionData):
        return Quantity(
            inch_to_cm(abs_to_inch(data.quantity.val)), unit=CentimeterUnit()
        )

    def to_inch(self, data: QuantityConversionData):
        return Quantity(abs_to_inch(data.quantity.val), unit=InchUnit())

    def to_relative(self, data: QuantityConversionData):
        return Quantity(
            val=data.quantity.val / data.length.to_points().val, unit=RelativeUnit()
        )

    def to_points(self, data: QuantityConversionData):
        return Quantity(val=data.val, unit=PointUnit())


class CentimeterUnit(UnitKind):
    str_type = "centimeter"
    is_length_unit = True

    def to_data(self, data: QuantityConversionData):
        new_val = (data.scale.high - data.scale.low) * data.quantity.to_relative(
            data.length, data.scale
        ).val
        return Quantity(val=new_val, unit=DataUnit())

    def to_centimeter(self, data: QuantityConversionData):
        return Quantity(data.quantity.val, unit=CentimeterUnit())

    def to_inch(self, data: QuantityConversionData):
        # TODO this has to be double checked, the rust code says inch to cm,
        # but logically this sounds like cm to inch
        return Quantity(inch_to_cm(data.quantity.val), unit=InchUnit())

    def to_relative(self, data: QuantityConversionData):
        new_val = data.quantity.to_points().val / data.length.to_points().val
        return Quantity(val=new_val, unit=RelativeUnit())

    def to_points(self, data: QuantityConversionData):
        return Quantity(
            val=inch_to_abs(cm_to_inch(data.quantity.val)), unit=PointUnit()
        )


class InchUnit(UnitKind):
    str_type = "inch"
    is_length_unit = True

    def to_data(self, data: QuantityConversionData):
        new_val = (data.scale.high - data.scale.low) * data.quantity.to_relative(
            data.length, data.scale
        ).val
        return Quantity(val=new_val, unit=DataUnit())

    def to_centimeter(self, data: QuantityConversionData):
        return Quantity(inch_to_cm(data.quantity.val), unit=CentimeterUnit())

    def to_inch(self, data: QuantityConversionData):
        return Quantity(data.quantity.val, unit=InchUnit())

    def to_relative(self, data: QuantityConversionData):
        new_val = data.quantity.to_points().val / data.length.to_points().val
        return Quantity(val=new_val, unit=RelativeUnit())

    def to_points(self, data: QuantityConversionData):
        return Quantity(val=inch_to_abs(data.val), unit=PointUnit())


class RelativeUnit(UnitKind):
    str_type = "relative"

    def to_data(self, data: QuantityConversionData):
        new_val = (data.scale.high - data.scale.low) * data.quantity.val
        return Quantity(val=new_val, unit=DataUnit())

    def to_relative(self, data: QuantityConversionData):
        return Quantity(val=data.quantity.val, unit=RelativeUnit())

    def to_points(self, data: QuantityConversionData):
        if data.length:
            return Quantity(val=data.val, unit=PointUnit())
        raise GGException("un expected")


class DataUnit(UnitKind):
    str_type = "data"

    def to_data(self, data: QuantityConversionData):
        return Quantity(val=data.quantity.val, unit=DataUnit())

    def to_relative(self, data: QuantityConversionData):
        if not data.scale:
            raise GGException(
                "Need a scale to convert quantity of kind Data to relative"
            )
        new_val = data.quantity.val / (data.scale.high - data.scale.low)
        return Quantity(val=new_val, unit=RelativeUnit())


class StrWidthUnit(UnitKind):
    str_type = "str_width"


class StrHeightUnit(UnitKind):
    str_type = "str_height"


def convert_quantity_data(
    kind: UnitKind, quantity: Quantity, data: ToQuantityData
) -> Quantity:
    if quantity.unit.str_type == kind.str_type:
        return quantity

    conversion_map = {
        "centimeter": lambda: quantity.to_centimeters(data),
        "point": lambda: quantity.to_points(data),
        "inch": lambda: quantity.to_inches(data),
        "data": lambda: quantity.to_data(data),
        "relative": lambda: quantity.to_relative(data),
    }

    func_ = conversion_map.get(kind.str_type)
    if not func_:
        raise GGException("conversion not implemented")

    return func_(ToQuantityData)


def unit_type_from_string(input_str):
    data = {
        "point": PointUnit,
        "centimeter": CentimeterUnit,
        "inch": InchUnit,
        "relative": RelativeUnit,
        "data": DataUnit,
        "str_widht": StrWidthUnit,
        "str_height": StrHeightUnit,
    }
    return data[input_str]
