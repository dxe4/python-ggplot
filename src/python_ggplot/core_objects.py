from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Callable, TypeVar, Generic
from python_ggplot.cairo_backend import CairoBackend
from python_ggplot.common import linspace

from python_ggplot.common import inch_to_abs, inch_to_cm, cm_to_inch, abs_to_inch


class MarkerKind(Enum):
    CIRCLE = auto()
    CROSS = auto()
    TRIANGLE = auto()
    RHOMBUS = auto()
    RECTANGLE = auto()
    ROTCROSS = auto()
    UPSIDEDOWN_TRIANGLE = auto()
    EMPTY_CIRCLE = auto()
    EMPTY_RECTANGLE = auto()
    EMPTY_RHOMBUS = auto()



class FileTypeKind(Enum):
    SVG = auto()
    PNG = auto()
    PDF = auto()
    VEGA = auto()
    TEX = auto()


@dataclass
class Image:
    fname: str
    width: int
    height: int
    ftype: FileTypeKind
    backend: CairoBackend


@dataclass
class HueConfig:
    hue_start: float = 15.0
    chroma: float = 100.0
    luminance: float = 65.0


class LineType(Enum):
    NONE_TYPE = auto()
    SOLID = auto()
    DASHED = auto()
    DOTTED = auto()
    DOT_DASH = auto()
    LONG_DASH = auto()
    TWO_DASH = auto()


class ErrorBarKind(Enum):
    LINES = auto()
    LINEST = auto()


class TextAlignKind(Enum):
    LEFT = auto()
    CENRTER = auto()
    RIGHT = auto()


class CFontSlant(Enum):
    NORMAL = auto()
    ITALIC = auto()
    OBLIQUE = auto()


@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float


@dataclass
class ColorHCL:
    h: float
    c: float
    l: float

    @staticmethod
    def gg_color_hue(num: int, hue_config: HueConfig) -> List["ColorHCL"]:
        hues = linspace(
            hue_config.hue_start, hue_config.hue_start + 360.0, num + 1, endpoint=True
        )
        colors = [
            ColorHCL(h=h, c=hue_config.chroma, l=hue_config.luminance) for h in hues
        ]
        return colors


@dataclass
class Font:
    family: str = "sans-serif"
    size: float = 12.0
    bold: bool = False
    slant: CFontSlant = CFontSlant.NORMAL
    color: Color = Color(r=0.0, g=0.0, b=0.0, a=1.0)
    align_kind: TextAlignKind = TextAlignKind.CENRTER


@dataclass
class Gradient:
    colors: List[Color]
    rotation: float


class AxisKind(Enum):
    X = auto()
    Y = auto()

T = TypeVar('T', int, float)


@dataclass
class Point(Generic[T]):
    x: T
    y: T


@dataclass
class Style:
    color: Color
    size: float
    line_type: LineType
    line_width: float
    fill_color: Color
    marker: MarkerKind
    error_bar_kind: ErrorBarKind
    gradient: Optional[Gradient]
    font: Font


class CompositeKind(Enum):
    ERROR_BAR = auto()


class TickKind(Enum):
    ONE_SIDE = auto()
    BOTH_SIDES = auto()


class OptionError(Exception):
    pass


def either(a, b):
    if a is not None:
        return a
    if b is not None:
        return b
    raise OptionError("Both options are None.")


@dataclass
class Scale:
    low: float
    high: float


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
            if self.length and self.quantity.unit in [PointUnit, CentimeterUnit, InchUnit]:
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


@dataclass
class ViewPort:
    name: str
    parent: str
    style: Style
    x_scale: Scale
    y_scale: Scale
    rotate: Optional[float] = None
    scale: Optional[float] = None
    origin: "Coord"
    width: Quantity
    height: Quantity
    objects: List["GraphicsObject"]
    children: List["ViewPort"]
    w_view: Quantity
    h_view: Quantity
    w_img: Quantity
    h_img: Quantity

    def apply_operator(
        self,
        other: "Quantity",
        length: Optional["Quantity"],
        scale: Optional[Scale],
        as_coordinate: bool,
        operator: Callable[[float, float], float],
    ) -> "Quantity":
        pass

    def point_height_height(self, dimension: Quantity) -> Quantity:

        if not isinstance(self.w_view.unit, PointUnit):
            raise ValueError(f"Expected Point, found {self.w_view.unit}")

        # Placeholder for relative width computation
        other = self.width.to_relative(dimension)
        return self.w_view.multiply(other)

    def point_width(self) -> Quantity:
        return self.point_height_height(self.w_view)

    def point_height(self) -> Quantity:
        return self.point_height_height(self.h_view)


def default_coord_view_location(view: ViewPort, kind: AxisKind):
    if kind == AxisKind.X:
        return view.point_width(), view.x_scale
    elif kind == AxisKind.Y:
        return view.point_height(), view.y_scale
    else:
        raise GGException("")


class GGException(Exception):
    pass


class UnitKind:

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

    def to_data(self, data: QuantityConversionData):
        new_val = (data.scale.high - data.scale.low) * data.quantity.to_relative(
            data.length, data.scale
        ).val
        return Quantity(val=new_val, unit=DataUnit())

    def to_centimeter(self, data: QuantityConversionData):
        return Quantity(inch_to_cm(abs_to_inch(data.quantity.val)), unit=CentimeterUnit())

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
    ) -> 'CoordType':
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
        return Quantity(val=inch_to_abs(cm_to_inch(data.quantity.val)), unit=PointUnit())

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


@dataclass
class LengthCoord:
    length: Optional[Quantity] = None

    def to_relative_coord1d(self, pos: float, length: Optional[Quantity], kind: UnitKind) -> 'Coord1D':
        length = self.length or length
        if length is None:
            raise ValueError("A length is required for relative conversion.")

        relative_length = length.to(kind)
        return Coord1D(pos / relative_length.val, RelativeCoordType())


@dataclass
class DataCoord:
    scale: Scale
    axis_kind: AxisKind

    def to_relative_coord1d(self, pos: float) -> Coord1D:
        if self.axis_kind == AxisKind.X:
            pos = (pos - self.scale.low) / (self.scale.high - self.scale.low)
        elif self.axis_kind == AxisKind.Y:
            pos = 1.0 - (pos - self.scale.low) / (self.scale.high - self.scale.low)
        else:
            raise ValueError("Invalid axis kind")

        return Coord1D(pos=pos, coord_type=RelativeCoordType())


@dataclass
class TextCoord:
    text: str
    font: Font

    def to_relative_coord1d(self, pos: float, coord_type: CoordType, length: Optional[Quantity]) -> Coord1D:
        # Get text dimensions
        text_extend = CairoBackend.get_text_extend(self.text, self.font)

        # this has to be str height or str width
        # todo add validation
        dimension = coord_type.text_extend_dimension(text_extend)

        if length is None:
            raise GGException("Conversion from StrWidth to relative requires a length scale!")

        pos = (pos * dimension) / length.to_points(None).val
        return Coord1D(pos=pos, coord_type=RelativeCoordType())


@dataclass
class CoordTypeConversion:
    coord1d: Coord1D
    length: Optional[Quantity] = None


@dataclass
class CoordType:
    is_length_coord = False

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        raise GGException("Not implemented")


@dataclass
class RelativeCoordType(CoordType):
    is_length_coord = False

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return Coord1D(pos=data.coord1d.pos, coord_type=RelativeCoordType())

@dataclass
class PointCoordType(CoordType):
    is_length_coord = True
    data: LengthCoord

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos, data.length, PointUnit)


@dataclass
class CentimeterCoordType(CoordType):
    is_length_coord = True
    data: LengthCoord

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos, data.length, CentimeterUnit)


@dataclass
class InchCoordType(CoordType):
    is_length_coord = True
    data: LengthCoord

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos, data.length, InchUnit)


@dataclass
class DataCoordType(CoordType):
    data: DataCoord

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos)



@dataclass
class StrWidthCoordType(CoordType):
    data: TextCoord

    def text_extend_dimension(self, text_extend):
        return text_extend.width()

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        return self.data.to_relative_coord1d(data.coord1d.pos, self, data.length)


@dataclass
class StrHeightCoordType(CoordType):
    data: TextCoord

    def text_extend_dimension(self, text_extend):
        return text_extend.height()

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        raise GGException("Not implemented")


@dataclass
class Coord1D:
    pos: float
    coord_type: CoordType
