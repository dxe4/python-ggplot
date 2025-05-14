import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from python_ggplot.core.coord.objects import Coord, Coord1D, coord_type_from_type
from python_ggplot.core.objects import (
    BLACK,
    GREY92,
    TRANSPARENT,
    AxisKind,
    Color,
    FileTypeKind,
    GGException,
    Image,
    LineType,
    MarkerKind,
    Point,
    Scale,
    Style,
    TextAlignKind,
)
from python_ggplot.core.units.objects import Quantity
from python_ggplot.graphics.cairo_backend import init_image
from python_ggplot.graphics.initialize import (
    CoordsInput,
    InitRectInput,
    InitTextInput,
    init_rect_from_coord,
    init_text,
)
from python_ggplot.graphics.objects import (
    GOAxis,
    GOGrid,
    GOLabel,
    GOLine,
    GOManyPoints,
    GOPoint,
    GOPolyLine,
    GORaster,
    GORect,
    GOText,
    GOTick,
    GOTickLabel,
    GOType,
    GraphicsObject,
    GraphicsObjectConfig,
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
            size=0.0,
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
    scale: Optional[Scale] = None,
    ignore_overflow: bool = False,
) -> list[Quantity]:
    zero_elems = sum(
        1
        for q in quantities
        if math.isclose(q.to_relative(scale=scale, length=length).val, 0.0)
    )

    if zero_elems == 0:
        return quantities

    width_sum = sum(q.to_relative(scale=scale, length=length).val for q in quantities)
    remain_width = (1.0 - width_sum) / float(zero_elems)

    # TODO URGENT: don't like 0.0 comparison, turn to int or use precision
    if not ignore_overflow and remain_width < 0.0:
        raise GGException(
            "given layout sizes exceed the viewport size."
            f"width sum: {width_sum} zero elms: {zero_elems}"
        )

    result: List[Quantity] = []
    for i in range(num):
        if math.isclose(quantities[i].to_relative(length=length, scale=scale).val, 0.0):
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
        return fill_empty_sizes_evenly(
            col_widths,
            view.point_width(),
            cols,
            scale=view.x_scale,
            ignore_overflow=ignore_overflow,
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
        return fill_empty_sizes_evenly(
            row_heights,
            view.point_height(),
            rows,
            scale=view.y_scale,
            ignore_overflow=ignore_overflow,
        )


def layout(
    view: ViewPort,
    cols: int,
    rows: int,
    col_widths: Optional[List[Quantity]] = None,
    row_heights: Optional[List[Quantity]] = None,
    margin: Optional[Quantity] = None,
    ignore_overflow: bool = False,
) -> None:
    margin = margin or Quantity.relative(0.0)

    if row_heights is None:
        row_heights = []

    if col_widths is None:
        col_widths = []

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
                    length=Quantity.points(view.point_width().val / cols),
                    scale=view.x_scale,
                    as_coordinate=False,
                ),
                length=view.point_width(),
                scale=view.x_scale,
                as_coordinate=False,
            )

            height: Quantity = heights[i].subtract(
                Quantity.relative(factor_h).multiply(
                    margin_y,
                    length=Quantity.points(view.point_height().val / rows),
                    scale=view.y_scale,
                    as_coordinate=False,
                ),
                length=view.point_height(),
                scale=view.y_scale,
                as_coordinate=False,
            )
            view_input: ViewPortInput = ViewPortInput(
                x_scale=view.x_scale,
                y_scale=view.y_scale,
                style=view.style,
                name=f"{view.name} - grid: ({i},{j})",
            )

            origin = Coord(x=current_col, y=current_row)
            child: ViewPort = view.add_viewport(origin, width, height, view_input)
            view.children.append(child)

            coord_cls = coord_type_from_type(widths[j].unit_type)
            current_col = current_col + coord_cls.from_view(
                view, AxisKind.X, widths[j].val
            )

        coord_cls = coord_type_from_type(heights[i].unit_type)
        current_row = current_row + coord_cls.from_view(
            view, AxisKind.Y, heights[i].val
        )


def background(view: ViewPort, style: Optional[Style] = None):
    default_style = Style(color=TRANSPARENT, fill_color=GREY92)

    style = style or default_style

    new_obj = GORect(
        name="background",
        config=GraphicsObjectConfig(style=style),
        origin=Coord.relative(0.0, 0.0),
        width=Quantity.relative(1.0),
        height=Quantity.relative(1.0),
    )
    view.objects.append(new_obj)


def draw_line(img: Image, gobj: Union[GOAxis, GOLine]):
    start = gobj.data.start.point()
    stop = gobj.data.stop.point()
    if gobj.config.style is None:
        raise GGException("expected style")

    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = (gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1])

    img.backend.draw_line(  # type: ignore
        img, start, stop, gobj.config.style, rotate_in_view
    )


def draw_rect(img: Image, gobj: GORect):
    if gobj.config.style is None:
        raise GGException("expected style")

    left = gobj.origin.point().x
    bottom = gobj.origin.point().y

    rotate = gobj.config.rotate
    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = (gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1])

    img.backend.draw_rectangle(  # type: ignore -> pycairo headache
        img=img,
        left=left,
        bottom=bottom,
        width=gobj.width.val,
        height=gobj.height.val,
        style=gobj.config.style,
        rotate=rotate,
        rotate_in_view=rotate_in_view,
    )


def draw_raster(img: Image, gobj: GORaster):
    left = gobj.origin.point().x
    bottom = gobj.origin.point().y

    if gobj.config.style is None:
        raise GGException("expected style")

    rotate = gobj.config.rotate
    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = (gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1])

    img.backend.draw_raster(  # type: ignore -> pycairo headache
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
    rotate_in_view: Optional[tuple[float, Point[float]]],
    marker: MarkerKind,
    angle: float,
    pos: Point[float],
    kind: Set[MarkerKind],
) -> Optional[Tuple[float, Point[float]]]:
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
    rotate_in_view: Optional[Tuple[float, Point[float]]] = None,
    style: Optional[Style] = None,
):
    if marker in (MarkerKind.CIRCLE, MarkerKind.EMPTY_CIRCLE):
        fill_color = color if marker == MarkerKind.CIRCLE else TRANSPARENT
        stroke_color = TRANSPARENT if marker == MarkerKind.CIRCLE else color

        img.backend.draw_circle(  # type: ignore -> pycairo headache
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
        img.backend.draw_line(img, start, stop, style, rotate)  # type: ignore -> pycairo headache

        # Draw vertical line
        start = Point(x=pos.x, y=pos.y - size)
        stop = Point(x=pos.x, y=pos.y + size)
        img.backend.draw_line(img, start, stop, style, rotate)  # type: ignore -> pycairo headache

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

        img.backend.draw_polyline(  # type: ignore -> pycairo headache
            img, points, style, rotate
        )

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

        img.backend.draw_rectangle(  # type: ignore -> pycairo headache
            img, left, bottom, size, size, style, None, rotate
        )


def draw_point(img: Image, gobj: GOPoint):
    pos = gobj.pos.point()
    style = gobj.config.style
    marker = gobj.marker
    size = gobj.size
    color = gobj.color

    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1]

    draw_point_impl(img, pos, marker, size, color, rotate_in_view, style)


def draw_many_points(img: Image, gobj: GOManyPoints):
    style = gobj.config.style
    marker = gobj.marker
    size = gobj.size
    color = gobj.color

    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1]

    for pos in gobj.pos:
        draw_point_impl(img, pos.point(), marker, size, color, rotate_in_view, style)


def draw_polyline(img: Image, gobj: GOPolyLine):
    if gobj.config.style is None:
        raise GGException("expected a style")

    points = [Point(x=coord.x.pos, y=coord.y.pos) for coord in gobj.pos]

    rotate_in_view = None
    if gobj.config.rotate_in_view:
        rotate_in_view = gobj.config.rotate_in_view[0], gobj.config.rotate_in_view[1]

    img_copy = img
    img_copy.backend.draw_polyline(  # type: ignore -> pycairo headache
        img, points, gobj.config.style, rotate_in_view
    )


def draw_text(img: Image, gobj: Union[GOText, GOLabel, GOTickLabel]):
    data = gobj.data
    text = data.text
    font = data.font
    at = data.pos.point()
    align_kind = data.align

    img_copy = img
    if not isinstance(text, str):
        # TODO we should process this earlier, by this point it should be a string
        # this is not really critical, the intention by now should have been to draw a string
        print(f"draw text was called with a type of {type(text)}")
        text = str(round(text, 2))

    img_copy.backend.draw_text(  # type: ignore -> pycairo headache
        img, text, font, at, align_kind, gobj.config.rotate, None
    )


def draw_tick(img: Image, gobj: GOTick):
    style = gobj.config.style
    if style is None:
        raise GGException("expected style")

    length = style.size

    if not gobj.major:
        length = length / 2.0
        style.line_width = style.line_width / 2.0

    if gobj.secondary:
        length = -length

    start, stop = gobj.get_start_stop_point(length)
    img.backend.draw_line(  # type: ignore -> pycairo headache
        img, start, stop, style, rotate_angle=gobj.config.rotate_in_view
    )


def draw_grid(img: Image, gobj: GOGrid):
    style = gobj.config.style
    if style is None:
        raise GGException("expected style")

    for x in gobj.x_pos:
        origin = gobj.origin
        origin_diag = gobj.origin_diagonal
        if origin is None or origin_diag is None:
            raise GGException("expected origin")

        start = Point(x=x.pos, y=origin.y.pos)
        stop = Point(x=x.pos, y=origin_diag.y.pos)

        img.backend.draw_line(  # type: ignore
            img,
            start,
            stop,
            style,
            rotate_angle=gobj.config.rotate_in_view,
        )

    for y in gobj.y_pos:
        origin = gobj.origin
        origin_diag = gobj.origin_diagonal
        if origin is None or origin_diag is None:
            raise GGException("expected origin")

        start = Point(x=origin.x.pos, y=y.pos)
        stop = Point(x=origin_diag.x.pos, y=y.pos)

        img.backend.draw_line(  # type: ignore -> pycairo headache
            img,
            start,
            stop,
            style,
            rotate_angle=gobj.config.rotate_in_view,
        )


def scale_point(point: Point[float], width: float, height: float) -> Point[float]:
    return Point(x=point.x * width, y=point.y * height)


def to_global_coords(gobj: GraphicsObject, img: Image):
    gobj.to_global_coords(img)


draw_lookup: Dict[GOType, Callable[..., Any]] = {
    # start/stop data
    GOType.LINE: draw_line,
    GOType.AXIS: draw_line,
    # text
    GOType.TEXT: draw_text,
    GOType.TICK_LABEL: draw_text,
    GOType.LABEL: draw_text,
    # others
    GOType.GRID_DATA: draw_grid,
    GOType.TICK_DATA: draw_tick,
    GOType.POINT_DATA: draw_point,
    GOType.MANY_POINTS_DATA: draw_many_points,
    GOType.POLYLINE_DATA: draw_polyline,
    GOType.RECT_DATA: draw_rect,
    GOType.RASTER_DATA: draw_raster,
}


def draw_graphics_object(img: Image, gobj: GraphicsObject):
    """
    TODO fix the type ignore
    we can do gobj.draw()
    but for now we prefer backwards compat
    """
    to_global_coords(gobj, img)
    if not gobj.go_type == GOType.COMPOSITE_DATA:
        draw_lookup[gobj.go_type](img, gobj)  # type: ignore


def transform_and_draw(
    img: Image, gobj: GraphicsObject, view: ViewPort, center_x: float, center_y: float
):
    if view.rotate:
        point = Point(x=center_x, y=center_y)
        scaled_point = scale_point(point, img.width, img.height)
        gobj.config.rotate_in_view = (view.rotate, scaled_point)

    gobj.embed_into(view)
    draw_graphics_object(img, gobj)


def draw_viewport(img: Image, view: ViewPort):
    center_x, center_y = view.get_center()

    for gobj in view.objects:
        transform_and_draw(img, gobj, view, center_x, center_y)
        for go_child in gobj.config.children:
            transform_and_draw(img, go_child, view, center_x, center_y)

    for view_child in view.children:
        embeded_view = view_child.embed_into(view)
        # todo implement quantity comparison, low priority
        if (view_child.h_img.val != view.h_img.val) or (
            view_child.w_img.val != view.w_img.val
        ):
            raise GGException("expected h_img and w_img to match")

        if view.rotate and embeded_view.rotate is None:
            embeded_view.rotate = view.rotate

        draw_viewport(img, embeded_view)


def draw_to_file(view: "ViewPort", filename: Union[str, Path]):
    # TODO PNG is hardcoded which is fine for now
    width = round(view.w_img.val)
    height = round(view.h_img.val)
    img = init_image(filename, width, height, FileTypeKind.PNG)
    draw_viewport(img, view)
    # Save the surface to a PNG file
    img.backend.canvas.write_to_png(filename)  # type: ignore
    del img
