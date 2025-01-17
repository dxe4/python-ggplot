from dataclasses import dataclass
from typing import List, Optional

from python_ggplot.core.coord.objects import Coord, Coord1D, coord_type_from_type
from python_ggplot.core.objects import (
    BLACK,
    TRANSPARENT,
    AxisKind,
    Color,
    GGException,
    LineType,
    Scale,
    Style,
    TextAlignKind,
)
from python_ggplot.core.units.objects import Quantity
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
