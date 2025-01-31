from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from python_ggplot.core.objects import Color, LineType, MarkerKind

if TYPE_CHECKING:
    from python_ggplot.gg.scales import ScaleType
    from python_ggplot.gg.types import GGStyle


@dataclass
class ScaleValue(ABC):

    def __eq__(self, value: object, /) -> bool:
        # TODO Critical
        # implement or fix logic in
        #  public_interface.common.scale_x_discrete_with_labels
        # for format_discrete_label_
        return super().__eq__(value)

    @abstractmethod
    def update_style(self, style: "GGStyle"):
        pass

    @property
    @abstractmethod
    def scale_type(self) -> ScaleType:
        pass


@dataclass
class TextScaleValue(ScaleValue):

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.TEXT


@dataclass
class SizeScaleValue(ScaleValue):
    size: Optional[float] = None

    def update_style(self, style: "GGStyle"):
        pass

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.SIZE


@dataclass
class ShapeScaleValue(ScaleValue):
    marker: Optional[MarkerKind] = None
    line_type: Optional[LineType] = None

    def update_style(self, style: "GGStyle"):
        style.marker = self.marker
        style.line_type = self.line_type

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.SHAPE


@dataclass
class AlphaScaleValue(ScaleValue):
    alpha: Optional[float] = None

    def update_style(self, style: "GGStyle"):
        pass

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.ALPHA


@dataclass
class FillColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    def update_style(self, style: "GGStyle"):
        style.fill_color = self.color
        style.color = self.color

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.FILL_COLOR


@dataclass
class ColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    def update_style(self, style: "GGStyle"):
        style.color = self.color

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.COLOR


@dataclass
class TransformedDataScaleValue(ScaleValue):
    val: Optional[Any] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.TRANSFORMED_DATA


@dataclass
class LinearDataScaleValue(ScaleValue):
    val: Optional[Any] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.LINEAR_DATA
