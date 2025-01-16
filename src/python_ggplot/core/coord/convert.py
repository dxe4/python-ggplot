from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, cast

from python_ggplot.core.common import DPI, abs_to_inch, inch_to_abs, inch_to_cm
from python_ggplot.core.coord.objects import (
    CentimeterCoordType,
    Coord1D,
    DataCoord,
    DataCoordType,
    InchCoordType,
    LengthCoord,
    PointCoordType,
    RelativeCoordType,
    StrHeightCoordType,
)
from python_ggplot.core.objects import AxisKind, Font, GGException, Scale, UnitType
from python_ggplot.core.units.objects import Quantity
from python_ggplot.graphics.cairo_backend import CairoBackend


def unit_to_point(kind: UnitType, pos):
    data = {
        UnitType.CENTIMETER: lambda: inch_to_abs(abs_to_inch(pos)),
        UnitType.POINT: lambda: pos,
        UnitType.INCH: lambda: inch_to_abs(pos),
    }
    convert = data.get(kind)
    if not convert:
        raise GGException("convert not possible")
    return convert()


@dataclass
class CoordConversionData:
    coord: "Coord1D"
    to_kind: UnitType = UnitType.RELATIVE
    length: Optional[Quantity] = None


@dataclass
class CoordViaPointData:
    coord: "Coord1D"
    to_kind: UnitType = UnitType.RELATIVE
    length: Optional[Quantity] = None
    abs_length: Optional[Quantity] = None
    axis: Optional[AxisKind] = None
    scale: Optional[Scale] = None
    text: Optional[str] = None
    font: Optional[Font] = None


def length_to_point(data: CoordConversionData):
    res_length = None
    if data.length:
        # todo sanity check right length is used
        res_length = data.length.to(UnitType.POINT)
    pos = unit_to_point(data.coord.unit_type, data.coord.pos)
    return PointCoordType(pos, LengthCoord(length=res_length))


def length_to_relative(data: CoordConversionData) -> Coord1D:
    length = data.coord.get_length() or data.length
    if length is None:
        raise ValueError("A length is required for relative conversion.")

    relative_length = length.to(data.coord.unit_type)
    return RelativeCoordType(data.coord.pos / relative_length.val)


def length_to_centimeter(data: CoordConversionData) -> Coord1D:
    pos = inch_to_cm(abs_to_inch(data.coord.pos))
    length = data.coord.get_length().to_centimeter()
    return CentimeterCoordType(pos, LengthCoord(length=length))


def length_to_inch(data: CoordConversionData) -> Coord1D:
    pos = abs_to_inch(data.coord.pos)
    length = data.coord.get_length().to_inch()
    return InchCoordType(pos, LengthCoord(length=length))


def coord_to_self(data: CoordConversionData):
    return deepcopy(data.coord)


def _to_relative_calculation(coord: DataCoordType):
    return (coord.pos - coord.data.scale.low) / (
        coord.data.scale.high - coord.data.scale.low
    )


def _to_relative_x(coord: DataCoordType):
    return _to_relative_calculation(coord)


def _to_relative_y(coord: DataCoordType):
    return 1.0 - _to_relative_calculation(coord)


def _to_relative_data(coord: DataCoordType):
    if coord.data.axis_kind == AxisKind.X:
        return _to_relative_x(coord)
    if coord.data.axis_kind == AxisKind.Y:
        return _to_relative_y(coord)
    raise GGException()


def data_to_point(data: CoordConversionData):
    return data.coord.to(UnitType.RELATIVE).to(UnitType.POINT, length=data.length)


def data_to_relative(data: CoordConversionData):
    data_coord = cast(DataCoordType, data.coord)
    return RelativeCoordType(pos=_to_relative_data(data_coord))


def text_to_point(data: CoordConversionData):
    if data.length is None:
        return GGException("length must be provided")

    if not isinstance(data.coord, (StrHeightCoordType, StrHeightCoordType)):
        raise GGException("unexpected")

    dimension = data.coord.point_dimension()
    pos = data.coord.pos * dimension
    return PointCoordType(pos, LengthCoord(length=data.length))


def text_to_relative(data: CoordConversionData):
    if data.length is None:
        raise GGException(
            "Conversion from StrWidth to relative requires a length scale!"
        )
    if not isinstance(data.coord, (StrHeightCoordType, StrHeightCoordType)):
        raise GGException("unexpected")

    text_extend = CairoBackend.get_text_extend(
        data.coord.data.text, data.coord.data.font
    )
    # this has to be str height or str width
    # todo add validation
    dimension = data.coord.text_extend_dimension(text_extend)
    pos = (data.coord.pos * dimension) / data.length.to(UnitType.POINT).val
    return RelativeCoordType(pos=pos)


def relative_to_point(data: CoordConversionData):
    if not data.length:
        raise GGException("expected length for conversion")

    return PointCoordType(
        data.coord.pos + data.length.val, LengthCoord(length=data.length)
    )


# via point conversion


def coord_to_self_via_point(data: CoordViaPointData):
    return deepcopy(data.coord)


def length_to_point_via_point(data: CoordViaPointData) -> Coord1D:
    return data.coord.to_points()


def length_to_centimeter_via_point(data: CoordViaPointData):
    new_pos = inch_to_cm(abs_to_inch(data.coord.to_points().pos))
    length = data.coord.get_length().to_centimeter()
    return PointCoordType(new_pos, data=LengthCoord(length=length))


def length_to_inch_via_point(data: CoordViaPointData):
    new_pos = abs_to_inch(data.coord.to_points().pos)
    length = data.coord.get_length().to_inch()
    return PointCoordType(new_pos, data=LengthCoord(length=length))


# via point relative conversion


def to_point_relative_via_point(data: CoordViaPointData):
    if data.abs_length is None:
        raise GGException("expected abs_length")

    return PointCoordType(
        data.coord.pos * data.abs_length.to_points().val / DPI,
        data=LengthCoord(length=data.abs_length.to_points()),
    )


def to_inch_relative_via_point(data: CoordViaPointData):
    if data.abs_length is None:
        raise GGException("expected abs_length")

    return InchCoordType(
        data.coord.pos * data.abs_length.to_points().val / DPI,
        data=LengthCoord(length=data.abs_length.to_inch()),
    )


def to_centimeter_relative_via_point(data: CoordViaPointData):
    if data.abs_length is None:
        raise GGException("expected abs_length")

    return CentimeterCoordType(
        data.coord.pos * data.abs_length.to_points().val / DPI,
        data=LengthCoord(length=data.abs_length.to_centimeter()),
    )


def to_data_relative_via_point(data: CoordViaPointData):
    if data.scale is None or data.axis is None:
        raise GGException("need to provide scale and axis")

    new_post = (data.scale.high - data.scale.low) * data.coord.pos + data.scale.low
    return DataCoordType(
        new_post, data=DataCoord(scale=data.scale, axis_kind=data.axis)
    )


lookup_table = {
    # INCH
    (UnitType.INCH, UnitType.CENTIMETER): length_to_centimeter,
    (UnitType.INCH, UnitType.INCH): coord_to_self,
    (UnitType.INCH, UnitType.RELATIVE): length_to_relative,
    (UnitType.INCH, UnitType.POINT): length_to_point,
    # CENTIMETER
    (UnitType.CENTIMETER, UnitType.CENTIMETER): coord_to_self,
    (UnitType.CENTIMETER, UnitType.INCH): length_to_inch,
    (UnitType.CENTIMETER, UnitType.RELATIVE): length_to_relative,
    (UnitType.CENTIMETER, UnitType.POINT): length_to_point,
    # POINT
    (UnitType.POINT, UnitType.CENTIMETER): length_to_centimeter,
    (UnitType.POINT, UnitType.INCH): length_to_inch,
    (UnitType.POINT, UnitType.RELATIVE): length_to_relative,
    (UnitType.POINT, UnitType.POINT): coord_to_self,
    # DATA
    (UnitType.DATA, UnitType.RELATIVE): data_to_relative,
    (UnitType.DATA, UnitType.DATA): coord_to_self,
    (UnitType.DATA, UnitType.POINT): data_to_point,
    # HIEGHT
    (UnitType.STR_HEIGHT, UnitType.RELATIVE): text_to_relative,
    (UnitType.STR_HEIGHT, UnitType.DATA): text_to_point,
    (UnitType.STR_HEIGHT, UnitType.STR_HEIGHT): coord_to_self,
    # WIDTH
    (UnitType.STR_WIDTH, UnitType.RELATIVE): text_to_relative,
    (UnitType.STR_WIDTH, UnitType.DATA): text_to_point,
    (UnitType.STR_WIDTH, UnitType.STR_WIDTH): coord_to_self,
    # RELATIVE
    (UnitType.RELATIVE, UnitType.RELATIVE): coord_to_self,
    (UnitType.RELATIVE, UnitType.POINT): relative_to_point,
}


via_point_lookup = {
    # INCH
    (UnitType.INCH, UnitType.CENTIMETER): length_to_centimeter_via_point,
    (UnitType.INCH, UnitType.INCH): coord_to_self_via_point,
    (UnitType.INCH, UnitType.POINT): length_to_point_via_point,
    # CENTIMETER
    (UnitType.CENTIMETER, UnitType.CENTIMETER): coord_to_self_via_point,
    (UnitType.CENTIMETER, UnitType.INCH): length_to_inch_via_point,
    (UnitType.CENTIMETER, UnitType.POINT): length_to_point_via_point,
    # POINT
    (UnitType.POINT, UnitType.CENTIMETER): length_to_centimeter_via_point,
    (UnitType.POINT, UnitType.INCH): length_to_inch_via_point,
    (UnitType.POINT, UnitType.POINT): coord_to_self_via_point,
    # DATA
    (UnitType.DATA, UnitType.DATA): coord_to_self_via_point,
    # HIEGHT
    (UnitType.STR_HEIGHT, UnitType.STR_HEIGHT): coord_to_self_via_point,
    # WIDTH
    (UnitType.STR_WIDTH, UnitType.STR_WIDTH): coord_to_self_via_point,
}
relative_via_point_lookup = {
    UnitType.RELATIVE: coord_to_self_via_point,
    UnitType.INCH: None,
    UnitType.POINT: None,
    UnitType.CENTIMETER: None,
    UnitType.DATA: None,
    UnitType.STR_HEIGHT: None,
    UnitType.STR_WIDTH: None,
}


def coord_convert_data(data: CoordConversionData, to_type: UnitType):
    if to_type == data.coord.unit_type:
        return deepcopy(data.coord)

    conversion_func = lookup_table.get((data.coord.unit_type, to_type))
    if not conversion_func:
        raise GGException("conversion not possible")

    return conversion_func(data)


def convert_coord(coord: Coord1D, to_type: UnitType, length=None):
    data = CoordConversionData(coord, length=length)
    return coord_convert_data(data, to_type)


def convert_via_point(
    coord: Coord1D,
    to_kind: UnitType,
    length=None,
    abs_length=None,
    scale=None,
    axis=None,
    text=None,
    font=None,
):
    if coord.unit_type == to_kind:
        return deepcopy(coord)

    data = CoordViaPointData(
        coord,
        to_kind=to_kind,
        length=length,
        abs_length=abs_length,
        scale=scale,
        axis=axis,
        text=text,
        font=font,
    )

    if coord.unit_type.is_length() and to_kind.is_length():
        func = via_point_lookup.get((data.coord.unit_type, to_kind))
        if not func:
            raise GGException("this should never happen")

        return func(data)
    else:
        rel = coord.to_relative()
        data = CoordViaPointData(
            rel,
            to_kind=to_kind,
            length=length,
            abs_length=abs_length,
            scale=scale,
            axis=axis,
            text=text,
            font=font,
        )
        func = relative_via_point_lookup.get(to_kind)
        if not func:
            raise GGException("conversion cannot be done")

        return func(data)
