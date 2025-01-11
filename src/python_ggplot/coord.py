from dataclasses import dataclass
from typing import Optional
import typing as tp

from python_ggplot.core_objects import AxisKind, Scale, GGException, Font
from python_ggplot.units import UnitKind, InchUnit, PointUnit, CentimeterUnit
from python_ggplot.cairo_backend import CairoBackend


if tp.TYPE_CHECKING:
    from python_ggplot.quantity import Quantity


@dataclass
class Coord1D:
    pos: float
    coord_type: "CoordType"


@dataclass
class CoordTypeConversion:
    coord1d: Coord1D
    length: Optional["Quantity"] = None
    backend: Optional["CairoBackend"] = None


@dataclass
class CoordType:
    is_length_coord = False

    def to_relative(self, data: CoordTypeConversion) -> Coord1D:
        raise GGException("Not implemented")


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
        return self.data.to_relative_coord1d(
            data.coord1d.pos, data.length, CentimeterUnit
        )


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
