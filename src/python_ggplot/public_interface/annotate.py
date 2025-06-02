import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from python_ggplot.common.enum_literals import LINE_TYPE_VALUES, MARKER_KIND_VALUES
from python_ggplot.core.chroma import str_to_color, to_opt_color
from python_ggplot.core.coord.objects import (
    Coord,
    Coord1D,
    DataCoord,
    DataCoordType,
    PointCoordType,
    StrHeightCoordType,
    TextCoordData,
)
from python_ggplot.core.objects import (
    TRANSPARENT,
    AxisKind,
    Color,
    Font,
    GGException,
    LineType,
    MarkerKind,
    Scale,
    Style,
    TextAlignKind,
)
from python_ggplot.core.units.objects import PointUnit, Quantity
from python_ggplot.gg.types import (
    PossibleColor,
    PossibleFloat,
    PossibleMarker,
    get_str_width,
    str_height,
)
from python_ggplot.graphics.initialize import (
    InitMultiLineInput,
    InitRectInput,
    init_multi_line_text,
    init_point,
    init_rect,
)
from python_ggplot.graphics.objects import (
    Curve,
    GOCurve,
    GOPoint,
    GOType,
    GraphicsObject,
    GraphicsObjectConfig,
)

if TYPE_CHECKING:
    from python_ggplot.gg.types import GgPlot
    from python_ggplot.graphics.views import ViewPort


class Annotation(ABC):

    @abstractmethod
    def get_graphics_objects(
        self, view: "ViewPort", plot: "GgPlot"
    ) -> List[GraphicsObject]:
        pass


@dataclass
class CurveAnnotation(Annotation):
    curve: Curve
    style: Style

    def get_graphics_objects(
        self, view: "ViewPort", plot: "GgPlot"
    ) -> List[GraphicsObject]:
        if view.x_scale is None or view.y_scale is None:
            raise GGException("expected x and y scale on view to draw a curve")

        go_poly_line = GOCurve.create(
            self.curve,
            view.x_scale,
            view.y_scale,
            name="curve_annotation",
            style=self.style,
        )
        return [go_poly_line]


@dataclass
class PointAnnotate(Annotation):
    x: float
    y: float
    style: Style

    def get_graphics_objects(
        self, view: "ViewPort", plot: "GgPlot"
    ) -> List[GraphicsObject]:
        if view.x_scale is None or view.y_scale is None:
            raise GGException("expected x and y scale on view to draw a curve")

        coord = Coord(
            x=Coord1D.create_data(self.x, view.x_scale, AxisKind.X),
            y=Coord1D.create_data(self.y, view.y_scale, AxisKind.Y),
        )
        go_point = GOPoint(
            name="point_annotation",
            config=GraphicsObjectConfig(style=self.style),
            marker=self.style.marker,
            size=self.style.size,
            color=self.style.color,
            pos=coord,
        )
        return [go_point]


@dataclass
class TextAnnotation(Annotation):
    left: Optional[float]
    bottom: Optional[float]
    right: Optional[float]
    top: Optional[float]
    x: Optional[float]
    y: Optional[float]
    text: str
    font: "Font"
    rotate: Optional[float]
    background_color: Color

    def calculate_position(
        self,
        start_pos: Optional[float],
        end_pos: Optional[float],
        data_pos: Optional[float],
        axis_kind: AxisKind,
        scale: Optional[Scale],
        view_length: Quantity,
        size: PointUnit,
        error_msg: str,
    ) -> float:
        if start_pos is not None:
            return Quantity.relative(start_pos).to_points(length=view_length).val
        elif end_pos is not None:
            return (
                Quantity.relative(end_pos)
                .to_points(length=view_length)
                .subtract(size)
                .val
            )
        else:
            if data_pos is None or scale is None:
                raise GGException(error_msg)

            return (
                DataCoordType(
                    pos=data_pos,
                    data=DataCoord(axis_kind=axis_kind, scale=scale),
                )
                .to_points(length=view_length)
                .pos
            )

    def get_left_bottom(
        self,
        view: "ViewPort",
        total_height: PointUnit,
        max_width: PointUnit,
    ) -> Tuple[float, float]:

        result_left = self.calculate_position(
            start_pos=self.left,
            end_pos=self.right,
            data_pos=self.x,
            axis_kind=AxisKind.X,
            scale=view.x_scale,
            view_length=view.point_width(),
            size=max_width,
            error_msg="expected annotation.x and view.x_scale",
        )

        result_bottom = self.calculate_position(
            start_pos=self.bottom,
            end_pos=self.top,
            data_pos=self.y,
            axis_kind=AxisKind.Y,
            scale=view.y_scale,
            view_length=view.point_height(),
            size=total_height,
            error_msg="expected annotation.y and view.y_scale",
        )

        return (result_left, result_bottom)

    def get_graphics_objects(
        self, view: "ViewPort", plot: "GgPlot"
    ) -> List[GraphicsObject]:
        ANNOT_RECT_MARGIN = 0.5
        rect_style = Style(
            fill_color=self.background_color, color=self.background_color
        )

        margin_h = StrHeightCoordType(
            pos=ANNOT_RECT_MARGIN,
            data=TextCoordData(text="W", font=self.font),
        ).to_points()

        margin_w = StrHeightCoordType(
            pos=ANNOT_RECT_MARGIN,
            data=TextCoordData(text="W", font=self.font),
        ).to_points()

        total_height: PointUnit = Quantity.points(
            str_height(self.text, self.font).val + (margin_h.pos * 2.0),
        )  # type: ignore

        font = self.font
        max_line = list(
            sorted(
                self.text.split("\n"),
                key=lambda x: get_str_width(x, font).val,
            )
        )[-1]
        max_width = get_str_width(max_line, font)

        rect_width = Quantity.points(
            max_width.val + margin_w.pos * 2.0,
        )
        left, bottom = self.get_left_bottom(view, total_height, max_width)

        rect_x = left - margin_w.pos
        rect_y = bottom - total_height.val + margin_h.pos

        graphics_objects = init_multi_line_text(
            view,
            origin=Coord(
                x=Coord1D.create_point(left, view.point_width()),
                y=Coord1D.create_point(bottom, view.point_height()),
            ),
            text=self.text,
            text_kind=GOType.TEXT,
            align_kind=TextAlignKind.LEFT,
            init_multi_line_input=InitMultiLineInput(
                rotate=self.rotate,
                font=self.font,
            ),
        )
        if self.background_color != TRANSPARENT:
            annot_rect = init_rect(
                view,
                origin=Coord(
                    x=PointCoordType(pos=rect_x), y=PointCoordType(pos=rect_y)
                ),
                width=rect_width,
                height=total_height,
                init_rect_input=InitRectInput(
                    style=rect_style, rotate=self.rotate, name="annotationBackground"
                ),
            )
            # background has to be drown first otherwise its drawn above the text
            graphics_objects.insert(0, annot_rect)

        return graphics_objects


def annotate_curve(
    x: Union[float, int],
    y: Union[float, int],
    xend: Union[float, int],
    yend: Union[float, int],
    curvature: Union[float, int] = 0.5,
    height_scale: Union[float, int] = 0.5,
    background_color: str = "black",
    size: Union[float, int] = 2.0,
    line_type: LINE_TYPE_VALUES = "solid",
    color: str = "black",
    alpha: float = 1.0,
    arrow: bool = False,
    arrow_size_percent: int = 8,
    arrow_angle: int = 25,
):
    color_ = str_to_color(color)
    if color_ is None:
        raise GGException(f"color: {color} not found")

    if not math.isclose(0.0, color_.a) and not math.isclose(alpha, color_.a):
        # if color="transparent" or color = Color(1,1,1,0)
        # do we override with alpha?
        # certainly not if alpha was teh default 1.0
        color_ = color_.update_with_copy(a=alpha)

    style = Style(
        line_width=size,
        line_type=LineType.eitem(line_type),
        # TODO allow both Color and str
        color=color_,
        fill_color=TRANSPARENT,
    )
    curve = Curve(
        x=x,
        y=y,
        xend=xend,
        yend=yend,
        curvature=curvature,
        arrow=arrow,
        arrow_size_percent=arrow_size_percent,
        arrow_angle=arrow_angle,
    )
    curve_annotation = CurveAnnotation(curve=curve, style=style)
    return curve_annotation


def annotate_text(
    text: str,
    left: Optional[float] = None,
    bottom: Optional[float] = None,
    right: Optional[float] = None,
    top: Optional[float] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    size: int = 12,
    emoji: bool = False,
    rotate: float = 0.0,
    background_color: str = "white",
) -> TextAnnotation:

    bg_color = to_opt_color(background_color)
    if bg_color is None:
        # TODO: implement hex (str) -> Color
        raise GGException(f"coulnd not convert {background_color} to color")

    if emoji:
        font_family = "Segoe UI Emoji"
    else:
        font_family = "sans-serif"
    result = TextAnnotation(
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        x=x,
        y=y,
        text=text,
        font=Font(size=size, family=font_family),
        rotate=rotate,
        background_color=bg_color,
    )

    if (result.x is None and result.left is None) or (
        result.y is None and result.bottom is None
    ):
        raise ValueError(
            "Both an x/left and y/bottom position has to be given to `annotate`!"
        )

    return result


def annotate_point(
    x: float | int,
    y: float | int,
    color: PossibleColor = "gray20",
    size: PossibleFloat = 2,
    marker: MARKER_KIND_VALUES = "circle",
    alpha: Optional[float] = 1.0,
) -> "PointAnnotate":

    color = to_opt_color(color)
    if alpha:
        color = color.update_with_copy(a=alpha)

    style = Style(
        color=color,
        size=size,
        marker=MarkerKind.eitem(marker),
    )
    return PointAnnotate(x=x, y=y, style=style)
