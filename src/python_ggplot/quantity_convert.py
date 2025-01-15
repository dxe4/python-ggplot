from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from python_ggplot.common import abs_to_inch, cm_to_inch, inch_to_abs, inch_to_cm
from python_ggplot.coord import (
    CentimeterCoordType,
    InchCoordType,
    LengthCoord,
    PointCoordType,
    RelativeCoordType,
)
from python_ggplot.core_objects import AxisKind, GGException, Scale, UnitType
from python_ggplot.units import (
    CentimeterUnit,
    DataUnit,
    InchUnit,
    PointUnit,
    RelativeUnit,
)

if TYPE_CHECKING:
    from python_ggplot.units import Quantity
    from python_ggplot.views import ViewPort


@dataclass
class QuantityConversionData:
    quantity: "Quantity"
    length: Optional["Quantity"] = None
    scale: Optional[Scale] = None

    def validate_generic_conversion(self, kind: UnitType):
        if kind == UnitType.DATA and not self.scale:
            raise GGException("cannot covnert to data without scale")

    def validate_to_relative_conversion(self):
        if self.quantity.unit_type not in [UnitType.DATA, UnitType.RELATIVE]:
            if self.length and self.quantity.unit_type in [
                UnitType.POINT,
                UnitType.CENTIMETER,
                UnitType.INCH,
            ]:
                raise GGException(
                    "length scale needed to convert quantity to relative value!"
                )

        elif (self.quantity.unit_type == UnitType.DATA) and not self.scale:
            raise GGException(
                "length scale needed to convert quantity to relative value!"
            )


@dataclass
class ToQuantityData:
    scale: Optional[Scale] = None
    length: Optional["Quantity"] = None


def point_to_data(data: QuantityConversionData):
    if data.scale is None:
        raise GGException("need to provide scale")

    new_val = (data.scale.high - data.scale.low) * data.quantity.to_relative(
        ToQuantityData(length=data.length, scale=data.scale)
    ).val
    return DataUnit(val=new_val)


def point_to_centimeter(data: QuantityConversionData):
    return CentimeterUnit(inch_to_cm(abs_to_inch(data.quantity.val)))


def point_to_inch(data: QuantityConversionData):
    return InchUnit(abs_to_inch(data.quantity.val))


def point_to_relative(data: QuantityConversionData):
    if data.length is None:
        raise GGException("need to provide length")
    return RelativeUnit(
        val=data.quantity.val / data.length.to_points(ToQuantityData()).val
    )


def point_to_point(data: QuantityConversionData):
    return deepcopy(data.quantity)


def centimeter_to_data(data: QuantityConversionData):
    if data.scale is None:
        raise GGException("need to provide scale")
    new_val = (data.scale.high - data.scale.low) * data.quantity.to_relative(
        ToQuantityData(length=data.length, scale=data.scale)
    ).val
    return DataUnit(val=new_val)


def centimeter_to_centimeter(data: QuantityConversionData):
    return deepcopy(data.quantity)


def centimeter_to_inch(data: QuantityConversionData):
    # TODO this has to be double checked, the rust code says inch to cm,
    # but logically this sounds like cm to inch
    return InchUnit(inch_to_cm(data.quantity.val))


def centimeter_to_relative(data: QuantityConversionData):
    if data.length is None:
        raise GGException("need to provide length")
    new_val = (
        data.quantity.to_points(ToQuantityData()).val
        / data.length.to_points(ToQuantityData()).val
    )
    return RelativeUnit(val=new_val)


def centimeter_to_point(data: QuantityConversionData):
    return PointUnit(val=inch_to_abs(cm_to_inch(data.quantity.val)))


def inch_to_data(data: QuantityConversionData):
    if data.scale is None:
        raise GGException("need to provide length")

    new_val = (data.scale.high - data.scale.low) * data.quantity.to_relative(
        ToQuantityData(length=data.length, scale=data.scale)
    ).val
    return DataUnit(val=new_val)


def inch_to_centimeter(data: QuantityConversionData):
    return CentimeterUnit(inch_to_cm(data.quantity.val))


def inch_to_inch(data: QuantityConversionData):
    return deepcopy(data.quantity)


def inch_to_relative(data: QuantityConversionData):
    if data.length is None:
        raise GGException("need to provide length")
    new_val = (
        data.quantity.to_points(ToQuantityData()).val
        / data.length.to_points(ToQuantityData()).val
    )
    return RelativeUnit(val=new_val)


def inch_to_points(data: QuantityConversionData):
    return PointUnit(val=inch_to_abs(data.quantity.val))


def relative_to_data(data: QuantityConversionData):
    if data.scale is None:
        raise GGException("need to provide a scale")
    new_val = (data.scale.high - data.scale.low) * data.quantity.val
    return DataUnit(val=new_val)


def relative_to_relative(data: QuantityConversionData):
    return RelativeUnit(val=data.quantity.val)


def relative_to_points(data: QuantityConversionData):
    if data.length:
        return PointUnit(val=data.quantity.val * data.length.val)
    raise GGException("un expected")


def data_to_data(data: QuantityConversionData):
    return deepcopy(data.quantity)


def data_to_relative(data: QuantityConversionData):
    if not data.scale:
        raise GGException("Need a scale to convert quantity of kind Data to relative")
    new_val = data.quantity.val / (data.scale.high - data.scale.low)
    return RelativeUnit(val=new_val)


lookup_table = {
    # RELATIVE
    (UnitType.RELATIVE, UnitType.RELATIVE): relative_to_relative,
    (UnitType.RELATIVE, UnitType.DATA): relative_to_data,
    (UnitType.RELATIVE, UnitType.POINT): relative_to_points,
    # INCH
    (UnitType.INCH, UnitType.DATA): inch_to_data,
    (UnitType.INCH, UnitType.CENTIMETER): inch_to_centimeter,
    (UnitType.INCH, UnitType.INCH): inch_to_inch,
    (UnitType.INCH, UnitType.RELATIVE): inch_to_relative,
    (UnitType.INCH, UnitType.POINT): inch_to_points,
    # CENTIMETER
    (UnitType.CENTIMETER, UnitType.DATA): centimeter_to_data,
    (UnitType.CENTIMETER, UnitType.CENTIMETER): centimeter_to_centimeter,
    (UnitType.CENTIMETER, UnitType.INCH): centimeter_to_inch,
    (UnitType.CENTIMETER, UnitType.RELATIVE): centimeter_to_relative,
    (UnitType.CENTIMETER, UnitType.POINT): centimeter_to_point,
    # CENTIMETER
    (UnitType.POINT, UnitType.DATA): point_to_data,
    (UnitType.POINT, UnitType.CENTIMETER): point_to_centimeter,
    (UnitType.POINT, UnitType.INCH): point_to_inch,
    (UnitType.POINT, UnitType.RELATIVE): point_to_relative,
    (UnitType.POINT, UnitType.POINT): point_to_point,
    # DATA
    (UnitType.DATA, UnitType.RELATIVE): data_to_relative,
    (UnitType.DATA, UnitType.DATA): data_to_data,
}


def convert_quantity_data(data: QuantityConversionData, to_type: UnitType):
    if to_type == data.quantity.unit_type:
        return deepcopy(data.quantity)

    conversion_func = lookup_table.get((data.quantity.unit_type, to_type))
    if not conversion_func:
        raise GGException("conversion not possible")

    data.validate_generic_conversion(to_type)
    return conversion_func(data)


def convert_quantity(quantity, to_type: UnitType, length=None, scale=None):
    data = QuantityConversionData(quantity=quantity, length=length, scale=scale)
    return convert_quantity_data(data, to_type)


def quantitiy_to_coord(quantity):
    conversion_data = {
        UnitType.RELATIVE: lambda: RelativeCoordType(quantity.pos),
        UnitType.POINT: lambda: PointCoordType(
            quantity.pos, LengthCoord(length=deepcopy(quantity))
        ),
        UnitType.INCH: lambda: InchCoordType(
            quantity.pos, LengthCoord(length=deepcopy(quantity))
        ),
        UnitType.CENTIMETER: lambda: CentimeterCoordType(
            quantity.pos, LengthCoord(length=deepcopy(quantity))
        ),
    }
    conversion = conversion_data[quantity.unit_type]
    return conversion


def to_relative_from_view(
    quantity: "Quantity", view: "ViewPort", axis: AxisKind
) -> "Quantity":
    if AxisKind.X == axis:
        length = view.point_width()
        scale = view.x_scale
        return convert_quantity(
            quantity, to_type=UnitType.RELATIVE, length=length, scale=scale
        )
    elif AxisKind.Y == axis:
        length = view.point_height()
        scale = view.y_scale
        return convert_quantity(
            quantity, to_type=UnitType.RELATIVE, length=length, scale=scale
        )
    raise GGException("unexpected")
