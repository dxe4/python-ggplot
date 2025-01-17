import math
from dataclasses import dataclass
from typing import List, Optional, Union, Set, Tuple

from python_ggplot.core.coord.objects import Coord, Coord1D, coord_type_from_type
from python_ggplot.core.objects import (
    BLACK,
    GREY92,
    TRANSPARENT,
    AxisKind,
    Color,
    GGException,
    LineType,
    Scale,
    Style,
    TextAlignKind,
    MarkerKind,
    Point,
    Image,
    LineType,
)
from python_ggplot.core.units.objects import Quantity
from python_ggplot.graphics.objects import (
    GORect,
    GORect,
    GOAxis,
    GOLine,
    GORaster,
    GraphicsObjectConfig,
)
from python_ggplot.graphics.initialize import (
    CoordsInput,
    InitRectInput,
    InitTextInput,
    init_rect_from_coord,
    init_text,
)
from python_ggplot.graphics.views import ViewPort, ViewPortInput


@dataclass
class DrawBoundaryInput:
    color: Optional["Color"] = None
    write_name: Optional[bool] = False
    write_number: Optional[int] = None
    style: Optional[Style] = None

    def get_style(self) -> Style:
        if self.style is not None:
            return self.style

        color = self.color or BLACK
        return Style(
            color=color,
            line_width=1.0,
            size=None,
            line_type=LineType.SOLID,
            fill_color=TRANSPARENT,
            marker=None,
            error_bar_kind=None,
            gradient=None,
            font=None,
        )


def draw_boundary(view: "ViewPort", draw_boundary_input: DrawBoundaryInput) -> None:
    style = draw_boundary_input.get_style()

    rect = init_rect_from_coord(
        view,
        InitRectInput(style=style),
        CoordsInput(
            left=0.0,
            bottom=0.0,
            width=1.0,
            height=1.0,
        ),
    )
    view.add_obj(rect)

    if draw_boundary_input.write_name:
        origin = Coord.relative(0.5, 0.5)

        data = InitTextInput(view.name, TextAlignKind.CENTER)
        text = init_text(view, origin, data)
        if text.config.style is None:
            text.config.style = view.style

        view.objects.append(text)

    if draw_boundary_input.write_number is not None:
        origin = Coord.relative(0.5, 0.5)

        data = InitTextInput(
            str(draw_boundary_input.write_number), TextAlignKind.CENTER
        )
        text = init_text(view, origin, data)
        if text.config.style is None:
            text.config.style = view.style

        view.objects.append(text)


def fill_empty_sizes_evenly(
    quantities: list[Quantity],
    length: Quantity,
    num: int,
    scale: Scale,
    ignore_overflow: bool = False,
) -> list[Quantity]:
    # TODO URGENT: don't like this, what if it's 0.0000000001?
    zero_elems = sum(
        1 for q in quantities if q.to_relative(scale=scale, length=length).val == 0.0
    )

    if zero_elems == 0:
        return quantities

    width_sum = sum(q.to_relative(scale=scale, length=length).val for q in quantities)
    remain_width = (1.0 - width_sum) / float(zero_elems)

    # TODO URGENT: don't like 0.0 comparison, turn to int or use precision
    if not ignore_overflow and remain_width < 0.0:
        raise GGException("given layout sizes exceed the viewport size.")

    result = []
    for i in range(num):
        # same here, convert to int
        if quantities[i].to_relative(length=length, scale=scale).val == 0.0:
            result.append(Quantity.relative(remain_width))
        else:
            result.append(quantities[i])

    return result


def _get_widths_for_layout(
    view: ViewPort, cols: int, col_widths: List[Quantity], ignore_overflow: bool = False
) -> List[Quantity]:
    if not col_widths:
        return [Quantity.relative(1.0 / cols)] * cols
    else:
        if view.x_scale is None:
            raise GGException("expected x scale")
        return fill_empty_sizes_evenly(
            col_widths, view.point_width(), cols, view.x_scale, ignore_overflow
        )


def _get_heights_for_layout(
    view: ViewPort,
    rows: int,
    row_heights: List[Quantity],
    ignore_overflow: bool = False,
) -> List[Quantity]:
    if not row_heights:
        return [Quantity.relative(1.0 / rows)] * rows
    else:
        if view.y_scale is None:
            raise GGException("expected x scale")

        return fill_empty_sizes_evenly(
            row_heights, view.point_height(), rows, view.y_scale, ignore_overflow
        )


def layout(
    view: ViewPort,
    cols: int,
    rows: int,
    col_widths: List[Quantity],
    row_heights: List[Quantity],
    margin: Optional[Quantity] = None,
    ignore_overflow: bool = False,
) -> None:
    margin = margin or Quantity.relative(0.0)

    if len(col_widths) != cols and col_widths:
        raise GGException("there must be one column width for each column")
    if len(row_heights) != rows and row_heights:
        raise GGException("there must be one row height for each row")

    widths: List[Quantity] = _get_widths_for_layout(
        view, cols, col_widths, ignore_overflow
    )
    heights: List[Quantity] = _get_heights_for_layout(
        view, rows, row_heights, ignore_overflow
    )

    current_row = Coord1D.create_relative(0.0)
    for i in range(rows):
        current_col = Coord1D.create_relative(0.0)
        for j in range(cols):
            margin_x = margin.to_relative(scale=view.x_scale, length=view.point_width())
            margin_y = margin.to_relative(
                scale=view.y_scale, length=view.point_height()
            )

            factor_w = 2.0
            factor_h = 2.0

            width: Quantity = widths[j].subtract(
                Quantity.relative(factor_w).multiply(
                    margin_x,
                    Quantity.points(view.point_width().val / cols),
                    view.x_scale,
                    False,
                ),
                view.point_width(),
                view.x_scale,
                False,
            )

            height: Quantity = heights[j].subtract(
                Quantity.relative(factor_h).multiply(
                    margin_y,
                    Quantity.points(view.point_height().val / cols),
                    view.x_scale,
                    False,
                ),
                view.point_height(),
                view.y_scale,
                False,
            )

            view_input: ViewPortInput = ViewPortInput(
                x_scale=view.x_scale, y_scale=view.y_scale, style=view.style, name=""
            )

            child: ViewPort = view.add_viewport(
                Coord(x=current_col, y=current_row), width, height, view_input
            )
            view.children.append(child)

            coord_cls = coord_type_from_type(widths[j].unit_type)
            current_col = coord_cls.from_view(view, AxisKind.X, widths[j].val)

        coord_cls = coord_type_from_type(widths[j].unit_type)
        current_row = coord_cls.from_view(view, AxisKind.Y, heights[i].val)


def background(view: ViewPort, style: Optional[Style] = None):
    default_style = Style(color=BLACK, fill_color=GREY92)

    style = style or default_style

    new_obj = GORect(
        name="",
        config=GraphicsObjectConfig(style=style),
        origin=Coord.relative(0.0, 0.0),
        width=Quantity.relative(1.0),
        height=Quantity.relative(1.0),
    )
    view.objects.append(new_obj)


def draw_line(img, gobj: Union[GOAxis, GOLine]):
    start = gobj.data.start.point()
    stop = gobj.data.stop.point()
    if gobj.config.style is None:
        raise GGException("expected style")

    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = (gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1])

    img.backend.draw_line(img, start, stop, gobj.config.style, rotate_in_view)


def draw_rect(img, gobj: GORect):
    left = gobj.origin.point().x
    bottom = gobj.origin.point().y

    if gobj.config.style is None:
        raise GGException("expected style")

    rotate = gobj.config.rotate
    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = (gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1])

    img.backend.draw_rectangle(
        img,
        left,
        bottom,
        gobj.width.val,
        gobj.height.val,
        gobj.config.style,
        rotate,
        rotate_in_view,
    )


def draw_raster(img, gobj: GORaster):
    left = gobj.origin.point().x
    bottom = gobj.origin.point().y

    if gobj.config.style is None:
        raise GGException("expected style")

    rotate = gobj.config.rotate
    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = (gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1])

    img.backend.draw_raster(
        img,
        left,
        bottom,
        gobj.pixel_width.val,
        gobj.pixel_height.val,
        gobj.block_x,
        gobj.block_y,
        gobj.draw_cb,
        rotate,
        rotate_in_view,
    )


def rotate_obj(
    rotate_in_view: Optional[tuple[float, Point]],
    marker: MarkerKind,
    angle: float,
    pos: Point,
    kind: Set[MarkerKind],
) -> Optional[Tuple[float, Point]]:
    if marker not in kind:
        return None

    if rotate_in_view:
        return (rotate_in_view[0] + angle, rotate_in_view[1])

    return (angle, pos)


def draw_point_impl(
    img: Image,
    pos: Point[float],
    marker: MarkerKind,
    size: float,
    color: Color,
    rotate_in_view: Optional[Tuple[float, Point]] = None,
    style: Optional[Style] = None,
):
    if marker in (MarkerKind.CIRCLE, MarkerKind.EMPTY_CIRCLE):
        fill_color = color if marker == MarkerKind.CIRCLE else TRANSPARENT
        stroke_color = TRANSPARENT if marker == MarkerKind.CIRCLE else color

        img.backend.draw_circle(
            img,
            pos,
            size,
            1.0,
            stroke_color if stroke_color else None,
            fill_color if fill_color else None,
            rotate_in_view,
        )

    if style is None:
        raise GGException("Expected a style")

    style.color = color
    style.line_width = size / 2.0
    style.line_type = LineType.SOLID
    style.fill_color = color

    if marker in (MarkerKind.CROSS, MarkerKind.ROTCROSS):
        rotate = rotate_obj(rotate_in_view, marker, 45.0, pos, {MarkerKind.ROTCROSS})
        rotate = (rotate[0], rotate[1]) if rotate else None

        # Draw horizontal line
        start = Point(x=pos.x - size, y=pos.y)
        stop = Point(x=pos.x + size, y=pos.y)
        img.backend.draw_line(img, start, stop, style, rotate)

        # Draw vertical line
        start = Point(x=pos.x, y=pos.y - size)
        stop = Point(x=pos.x, y=pos.y + size)
        img.backend.draw_line(img, start, stop, style, rotate)

    elif marker in (MarkerKind.TRIANGLE, MarkerKind.EMPTY_RECTANGLE):
        # TODO sanity check this
        step = math.sin(math.radians(60)) * size
        rotate = rotate_obj(
            rotate_in_view, marker, 180.0, pos, {MarkerKind.UPSIDEDOWN_TRIANGLE}
        )
        rotate = (rotate[0], rotate[1]) if rotate else None

        points = [
            Point(x=pos.x - step, y=pos.y + step),  # bottom left
            Point(x=pos.x, y=pos.y - step),  # top middle
            Point(x=pos.x + step, y=pos.y + step),  # bottom right
        ]

        img.backend.draw_polyline(img, points, style, rotate)

    elif marker in (
        MarkerKind.RHOMBUS,
        MarkerKind.RECTANGLE,
        MarkerKind.UPSIDEDOWN_TRIANGLE,
        MarkerKind.EMPTY_RHOMBUS,
    ):
        fill_color = (
            color
            if marker in (MarkerKind.RECTANGLE, MarkerKind.RHOMBUS)
            else TRANSPARENT
        )

        rotate = rotate_obj(
            rotate_in_view,
            marker,
            45.0,
            pos,
            {MarkerKind.RHOMBUS, MarkerKind.EMPTY_RHOMBUS},
        )
        rotate = (rotate[0], rotate[1]) if rotate else None

        size = size * 1.5
        left = pos.x - (size / 2.0)
        bottom = pos.y - (size / 2.0)

        img.backend.draw_rectangle(img, left, bottom, size, size, style, None, rotate)
