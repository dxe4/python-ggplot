from dataclasses import dataclass
from typing import Optional
from python_ggplot.core_objects import GGException, AxisKind, ViewPort
from python_ggplot.coord import (
    PointCoordType,
    LengthCoord,
    CoordType,
    DataCoord,
    DataCoordType,
    RelativeCoordType,
)
from python_ggplot.common import inch_to_abs, inch_to_cm, abs_to_inch, cm_to_inch


def default_coord_view_location(view: ViewPort, kind: AxisKind):
    if kind == AxisKind.X:
        return view.point_width(), view.x_scale
    elif kind == AxisKind.Y:
        return view.point_height(), view.y_scale
    else:
        raise GGException("")


@dataclass
class QuantityConversionData:
    quantity: "Quantity"
    length: Optional["Quantity"] = None
    scale: Optional["Scale"] = None

    def validate_generic_conversion(self, kind):
        if kind == self.quantity.__class__:
            # TODO remove __class__ usage
            return Quantity(val=self.quantity.val, unit=self.quantity.unit)
        if kind == DataUnit and not self.scale:
            # TODO remove __class__ usage
            raise GGException("cannot covnert to data without scale")

    def validate_to_relative_conversion(self):
        # todo refactor
        if self.quantity.unit.__class__ not in [DataUnit, RelativeUnit]:
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

    def to_data(self, scale: "Scale", length: "Quantity"):
        data = QuantityConversionData(quantity=self, length=length, scale=scale)
        return self.unit.to_data(self, data)

    def to_centimeter(self):
        data = QuantityConversionData(quantity=self)
        return self.unit.to_centimeter(data)

    def to_inch(self):
        data = QuantityConversionData(quantity=self)
        return self.unit.to_inch(data)

    def to_points(self, length=None):
        data = QuantityConversionData(quantity=self, length=length)
        return self.unit.to_points(data)

    def to_relative(self, length=None, scale=None):
        data = QuantityConversionData(quantity=self, length=length, scale=scale)
        data.validate_to_relative_conversion()
        return self.unit.to_relative(data)


class UnitKind:
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

    def default_length_and_scale(self, view: ViewPort, kind: "UnitKind"):
        length, scale = default_coord_view_location(view, kind)
        return length, scale


class PointUnit(UnitKind):
    def is_point(self):
        # todo temp hack
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

    def create_default_coord_type(
        self, view: ViewPort, at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> "CoordType":
        length, _ = super().default_length_and_scale(view, kind)
        return PointCoordType(data=LengthCoord(length=length.to_points()))


class CentimeterUnit(UnitKind):
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

    def create_default_coord_type(
        self, view: ViewPort, at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> CoordType:
        length, _ = super().default_length_and_scale(view, kind)
        return PointCoordType(data=LengthCoord(length=length.to_centimeter()))


class InchUnit(UnitKind):
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

    def create_default_coord_type(
        self, view: ViewPort, at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> CoordType:
        length, _ = super().default_length_and_scale(view, kind)
        return PointCoordType(data=LengthCoord(length=length.to_inch()))


class RelativeUnit(UnitKind):
    def to_data(self, data: QuantityConversionData):
        new_val = (data.scale.high - data.scale.low) * data.quantity.val
        return Quantity(val=new_val, unit=DataUnit())

    def to_relative(self, data: QuantityConversionData):
        return Quantity(val=data.quantity.val, unit=RelativeUnit())

    def to_points(self, data: QuantityConversionData):
        if data.length:
            return Quantity(val=data.val, unit=PointUnit())
        raise GGException("un expected")

    def create_default_coord_type(
        self, view: ViewPort, at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> CoordType:
        return RelativeCoordType()


class DataUnit(UnitKind):

    def to_data(self, data: QuantityConversionData):
        return Quantity(val=data.quantity.val, unit=DataUnit())

    def to_relative(self, data: QuantityConversionData):
        if not data.scale:
            raise GGException(
                "Need a scale to convert quantity of kind Data to relative"
            )
        new_val = data.quantity.val / (data.scale.high - data.scale.low)
        return Quantity(val=new_val, unit=RelativeUnit())

    def create_default_coord_type(
        self, view: ViewPort, at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> CoordType:
        _, scale = super().default_length_and_scale(view, kind)
        data = DataCoord(scale=scale, axis_kind=axis_kind)
        return DataCoordType(data=data)


class StrWidthUnit(UnitKind):

    def create_default_coord_type(
        self, view: ViewPort, at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> CoordType:
        raise GGException("not implemented")


class StrHeightUnit(UnitKind):

    def create_default_coord_type(
        self, view: ViewPort, at: float, axis_kind: AxisKind, kind: UnitKind
    ) -> CoordType:
        raise GGException("not implemented")
