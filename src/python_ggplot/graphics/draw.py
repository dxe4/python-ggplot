from dataclasses import dataclass
from typing import Optional

from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import (
    BLACK,
    TRANSPARENT,
    Color,
    LineType,
    Style,
    TextAlignKind,
)
from python_ggplot.graphics.initialize import (
    CoordsInput,
    InitRectInput,
    InitTextInput,
    init_rect_from_coord,
    init_text,
)
from python_ggplot.graphics.views import ViewPort


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
