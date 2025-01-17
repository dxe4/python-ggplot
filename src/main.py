# todo, move this into unit tests eventually
import math

from python_ggplot.core.common import linspace
from python_ggplot.core.coord.objects import (
    CentimeterCoordType,
    Coord,
    Coord1D,
    CoordsInput,
)
from python_ggplot.core.objects import (
    AxisKind,
    Color,
    ErrorBarKind,
    MarkerKind,
    Point,
    Scale,
    Style,
)
from python_ggplot.core.units.objects import Quantity
from python_ggplot.graphics.cairo_backend import cairo_test
from python_ggplot.graphics.draw import background, draw_to_file
from python_ggplot.graphics.initialize import (
    InitAxisInput,
    InitErrorBarData,
    InitRectInput,
    TickLabelsInput,
    init_axis,
    init_error_bar_from_point,
    init_grid_lines,
    init_point_without_style,
    init_poly_line_from_points,
    init_rect,
    init_rect_from_coord,
    tick_labels,
    xlabel_from_float,
    xticks,
    ylabel_from_float,
    yticks,
)
from python_ggplot.graphics.views import ViewPort, ViewPortInput

# import numpy as np
# from dataclasses import dataclass
# from typing import Optional, List


TAU = 6.28318530717958647692528676655900577


def main():
    cairo_test()

    input_viewport = ViewPortInput()
    view = ViewPort.from_coords(CoordsInput(), input_viewport)
    coords_input = CoordsInput(left=0.1, bottom=0.1, width=0.8, height=0.8)
    input_viewport = ViewPortInput(
        x_scale=Scale(low=0.0, high=2.0 * math.pi), y_scale=Scale(low=-1.0, high=1.0)
    )
    view1 = view.add_viewport_from_coords(coords_input, input_viewport)

    coords_input2 = CoordsInput(left=0.25, bottom=0.5, width=0.75, height=0.5)
    input_viewport2 = ViewPortInput(
        x_scale=Scale(low=0.0, high=2.0 * math.pi), y_scale=Scale(low=-1.0, high=1.0)
    )
    view2 = ViewPort.from_coords(coords_input2, input_viewport2)
    line1 = init_axis(AxisKind.X, InitAxisInput())
    line2 = init_axis(AxisKind.Y, InitAxisInput())

    x = linspace(0.0, TAU, 10)
    y = [math.sin(i) for i in x]
    points = [Point(x=x_val, y=y_val) for x_val, y_val in zip(x, y)]
    gobj_points = []
    gobj_errors = []

    for point in points:
        pos = Coord(
            x=Coord1D.create_relative(point.x), y=Coord1D.create_relative(point.y)
        )
        new_point = init_point_without_style(
            pos=pos,
            size=None,
            marker=MarkerKind.CROSS,
            color=None,
            name=None,
        )
        gobj_points.append(new_point)

        init_error_data1 = InitErrorBarData(
            view=view2,
            point=point,
            error_up=CentimeterCoordType.from_view(view2, AxisKind.X, 0.5),
            error_down=CentimeterCoordType.from_view(view2, AxisKind.X, 0.5),
            axis_kind=AxisKind.X,
            error_bar_kind=ErrorBarKind.LINES,
            style=None,
            name=None,
        )
        init_error_data2 = InitErrorBarData(
            view=view2,
            point=point,
            error_up=CentimeterCoordType.from_view(view2, AxisKind.X, 0.25),
            error_down=CentimeterCoordType.from_view(view2, AxisKind.X, 0.25),
            axis_kind=AxisKind.X,
            error_bar_kind=ErrorBarKind.LINES,
            style=None,
            name=None,
        )
        error1 = init_error_bar_from_point(init_error_data1)
        error2 = init_error_bar_from_point(init_error_data2)
        gobj_errors.extend([error1, error2])

    points_line = init_poly_line_from_points(view1, points)
    x_ticks = xticks(view1, [])
    y_ticks = yticks(view1, [])
    x_tick_labels = tick_labels(view1, x_ticks, TickLabelsInput())
    y_tick_labels = tick_labels(view1, y_ticks, TickLabelsInput())

    coords_input = CoordsInput(left=0.3, bottom=0.3, width=0.2, height=0.1)
    rect = init_rect_from_coord(view1, InitRectInput(), coords_input)
    color = Color(r=1.0, g=0.0, b=0.0, a=0.0)
    fill_color = Color(r=0.0, g=0.0, b=0.0, a=0.5)
    data = InitRectInput(
        style=Style(line_width=2.0, color=color, fill_color=fill_color)
    )
    rect2 = init_rect_from_coord(view2, data, coords_input)
    origin = Coord.relative(0.1, 0.1)

    width = Quantity.centimeters(1.0)
    height = Quantity.centimeters(1.0)
    cm_square = init_rect(view1, origin, width, height, InitRectInput())

    origin = Coord.relative(0.3, 0.3)
    inch_square = init_rect(view1, origin, width, height, InitRectInput())

    x_label = xlabel_from_float(view1, "Energy")
    y_label = ylabel_from_float(view1, "Count")

    background(view)

    grid_lines = init_grid_lines(x_ticks=x_ticks, y_ticks=y_ticks)

    grid_lines_major = init_grid_lines(x_ticks=x_ticks, y_ticks=y_ticks, major=False)

    objects = [
        x_ticks,
        y_ticks,
        x_tick_labels,
        y_tick_labels,
        [
            line1,
            line2,
            x_label,
            y_label,
            cm_square,
            grid_lines,
            grid_lines_major,
            points_line,
            inch_square,
        ],
    ]
    for object_vec in objects:
        for obj in object_vec:
            view1.objects.append(obj)

    objects = [[rect2], gobj_errors, gobj_points]
    for object_vec in objects:
        for obj in object_vec:
            view2.objects.append(obj)

    view1.children.append(view2)
    view.children.append(view1)
    draw_to_file(view, "simple_test.png")


if __name__ == "__main__":
    main()
