# todo this has to be split up into multiple files
import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

from python_ggplot.core.common import linspace, nice_number
from python_ggplot.core.coord.objects import (
    Coord,
    Coord1D,
    CoordsInput,
    coord_quantity_add,
    coord_quantity_sub,
    path_coord_view_port,
)
from python_ggplot.core.objects import BLACK  # GREY20,; GREY92,;
from python_ggplot.core.objects import (
    TRANSPARENT,
    WHITE,
    AxisKind,
    Color,
    CompositeKind,
    ErrorBarKind,
    Font,
    GGException,
    GOGrid,
    Gradient,
    LineType,
    MarkerKind,
    Point,
    Scale,
    Style,
    TextAlignKind,
    TickKind,
    UnitType,
)
from python_ggplot.core.units.objects import CentimeterUnit, Quantity
from python_ggplot.graphics.objects import (
    GOAxis,
    GOComposite,
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
    StartStopData,
    TextData,
    format_tick_value,
)
from python_ggplot.graphics.views import ViewPort, x_axis_y_pos, y_axis_x_pos


@dataclass
class InitAxisInput:
    width: float = 1.0
    color: Color = field(default_factory=lambda: Color(r=0.0, g=0.0, b=0.0, a=1.0))


@dataclass
class InitRasterData:
    callback: Callable[[], List[int]]
    num_x: int
    num_y: int
    rotate: Optional[float] = None
    name: Optional[str] = None


@dataclass
class InitTextInput:
    text: str = ""
    align_kind: TextAlignKind = TextAlignKind.CENTER
    font: Optional[Font] = None
    rotate: Optional[float] = None
    name: Optional[str] = None

    @classmethod
    def new(cls, text: str, align_kind: TextAlignKind) -> "InitTextInput":
        return cls(text=text, align_kind=align_kind)


@dataclass
class InitRectInput:
    color: Optional[Color] = field(
        default_factory=lambda: Color(r=0.0, g=0.0, b=0.0, a=1.0)
    )
    gradient: Optional[Gradient] = None
    style: Optional[Style] = None
    rotate: Optional[float] = None
    name: str = "rect"


@dataclass
class InitMultiLineInput:
    font: Optional[Font] = None
    rotate: Optional[float] = None
    name: str = "multi_line_text"


@dataclass
class InitLineInput:
    style: Optional[Style] = None
    name: Optional[str] = "line"


@dataclass
class InitErrorBarData:
    view: ViewPort
    point: Union[Coord, Point[float]]
    error_up: Coord1D
    error_down: Coord1D
    axis_kind: AxisKind
    error_bar_kind: ErrorBarKind
    style: Optional[Style] = None
    name: Optional[str] = None

    def create_lines_input(self):
        data = {
            AxisKind.X: {
                "x1": self.error_up,
                "x2": self.error_down,
                "y1": self.point.y,
                "y2": self.point.y,
                "style": self.style,
            },
            AxisKind.Y: {
                "x1": self.point.x,
                "x2": self.point.x,
                "y1": self.error_up,
                "y2": self.error_down,
                "style": self.style,
            },
        }
        return data[self.axis_kind]


def _init_axis_data(axis: AxisKind):
    if axis == AxisKind.X:
        name = "x_axis"
        start = Coord.relative(0.0, 1.0)
        stop = Coord.relative(1.0, 1.0)
        return start, stop, name
    else:
        name = "y_axis"
        start = Coord.relative(0.0, 0.0)
        stop = Coord.relative(0.0, 1.0)
        return start, stop, name


def init_axis(axis_kind: AxisKind, init_axis_input: InitAxisInput) -> GraphicsObject:
    start, stop, name = _init_axis_data(axis_kind)
    start_stop_data = StartStopData(start=start, stop=stop)
    style = Style(
        line_width=init_axis_input.width,
        color=init_axis_input.color,
    )

    graphics_obj = GOAxis(
        name=name,
        config=GraphicsObjectConfig(style=style),
        data=start_stop_data,
    )

    return graphics_obj


def init_rect(
    view: ViewPort,
    origin: Coord,
    width: Quantity,
    height: Quantity,
    init_rect_input: InitRectInput,
) -> GraphicsObject:
    style = init_rect_input.style or Style(
        gradient=init_rect_input.gradient,
        fill_color=init_rect_input.color,
        color=BLACK,
        line_width=0.0,
        size=0.0,
    )

    return GORect(
        name=init_rect_input.name,
        config=GraphicsObjectConfig(style=style),
        origin=path_coord_view_port(origin, view),
        width=width,
        height=height,
    )


def init_rect_from_coord(
    view: ViewPort, init_rect_input: InitRectInput, coords_input: CoordsInput
) -> GraphicsObject:
    origin = Coord(
        x=Coord1D.create_relative(coords_input.left),
        y=Coord1D.create_relative(coords_input.bottom),
    )
    width = Quantity.relative(coords_input.width)
    height = Quantity.relative(coords_input.height)

    return init_rect(view, origin, width, height, init_rect_input)


def init_raster(view, origin, width, height, init_raster_data: InitRasterData):
    return GORaster(
        name=init_raster_data.name or "raster",
        config=GraphicsObjectConfig(),
        origin=origin.path_coord_view_port(view),
        pixel_width=width,
        pixel_height=height,
        block_x=init_raster_data.num_x,
        block_y=init_raster_data.num_y,
        draw_cb=init_raster_data.callback,
        rotate=init_raster_data.rotate,
    )


def init_text(view: ViewPort, origin: Coord, init_text_data: InitTextInput) -> GOText:
    data = TextData(
        text=init_text_data.text,
        font=init_text_data.font or Font(),
        pos=path_coord_view_port(origin, view),
        align=init_text_data.align_kind,
    )
    return GOText(
        name=init_text_data.name or "text",
        config=GraphicsObjectConfig(
            rotate=init_text_data.rotate,
        ),
        data=data,
    )


def init_label(view: ViewPort, origin: Coord, init_text_data: InitTextInput) -> GOLabel:
    data = TextData(
        text=init_text_data.text,
        font=init_text_data.font or Font(),
        pos=path_coord_view_port(origin, view),
        align=init_text_data.align_kind,
    )
    return GOLabel(
        name=init_text_data.name or "text",
        config=GraphicsObjectConfig(rotate=init_text_data.rotate),
        data=data,
    )


def init_tick_label(
    view: ViewPort, origin: Coord, init_text_data: InitTextInput
) -> GOTickLabel:
    data = TextData(
        text=init_text_data.text,
        font=init_text_data.font or Font(),
        pos=path_coord_view_port(origin, view),
        align=init_text_data.align_kind,
    )
    return GOTickLabel(
        name=init_text_data.name or "text",
        config=GraphicsObjectConfig(rotate=init_text_data.rotate),
        data=data,
    )


init_text_lookup = {
    GOType.TEXT: init_text,
    GOType.TICK_LABEL: init_tick_label,
    GOType.LABEL: init_label,
}


def init_multi_line_text(
    view: ViewPort,
    origin: Coord,
    text: str,
    text_kind: GOType,
    align_kind: TextAlignKind,
    init_multi_line_input: InitMultiLineInput,
) -> None:
    if text_kind not in (GOType.TEXT, GOType.LABEL, GOType.TICK_LABEL):
        raise GGException("unexpected graphic object kind")

    font = init_multi_line_input.font or Font()
    lines = text.split("\n")
    lines_len = len(lines)

    for idx, line in enumerate(lines):
        pos = lines_len - idx - 0.5
        new_y = origin.y - Coord1D.create_str_height(pos, font).to_relative(
            view.point_height()
        )
        new_origin = Coord(x=origin.x, y=new_y)

        init_text_data = InitTextInput(
            text=line,
            align_kind=align_kind,
            font=font,
            rotate=init_multi_line_input.rotate,
            name=f"{init_multi_line_input.name} {idx}",
        )

        func = init_text_lookup[text_kind]
        res = func(view, new_origin, init_text_data)
        view.objects.append(res)


def init_line(
    start: Coord, stop: Coord, init_line_input: InitLineInput
) -> GraphicsObject:
    start_stop_data = {"start": start, "stop": stop}

    default_style = Style(
        color=BLACK,
        line_width=1.0,
    )

    return GOLine(
        name=init_line_input.name or "line",
        config=GraphicsObjectConfig(style=init_line_input.style or default_style),
        data=start_stop_data,
    )


def init_point(pos: Coord, style: Style, name: Optional[str]) -> GraphicsObject:
    if not style.size or not style.marker:
        raise GGException("expected size and market on style")

    return GOPoint(
        name=name or "line",
        config=GraphicsObjectConfig(style=style),
        marker=style.marker,
        pos=pos,
        size=style.size,
        color=style.color,
    )


def init_point_without_style(pos, size=None, marker=None, color=None, name=None):
    style = Style(
        marker=marker or MarkerKind.CIRCLE, size=size or 3.0, color=color or BLACK
    )
    return init_point(pos, style, name)


def init_point_from_point(
    view: ViewPort, pos, size=None, marker=None, color=None, name=None
):

    if view.x_scale is None or view.y_scale is None:
        raise GGException("x and y scale need to be setup")

    style = Style(
        marker=marker or MarkerKind.CIRCLE, size=size or 3.0, color=color or BLACK
    )

    pos = Coord(
        x=Coord1D.create_data(pos.x, view.x_scale, AxisKind.X),
        y=Coord1D.create_data(pos.y, view.y_scale, AxisKind.Y),
    )

    return init_point(pos, style, name)


def init_many_points(pos: List[float], style: Style, name: Optional[str] = None):
    return GOManyPoints(
        name=name or "many_points",
        config=GraphicsObjectConfig(style=style),
        marker=style.marker,
        pos=pos,
        size=style.size,
        color=style.color,
    )


def is_scale_not_trivial(coord: Coord1D):
    if coord.unit_type != UnitType.DATA:
        raise GGException("needed coord with data")

    scale = coord.get_scale()
    return scale.low != scale.high


def create_lines(
    x1: Coord1D, x2: Coord1D, y1: Coord1D, y2: Coord1D, style: Optional[Style] = None
) -> GraphicsObject:
    start = Coord(x=x1, y=y1)
    stop = Coord(x=x2, y=y2)
    data = InitLineInput(style=style)
    return init_line(start, stop, data)


def init_error_bar(data: InitErrorBarData) -> GraphicsObject:
    # TODO this code can improve
    if not isinstance(data.point, Coord):
        raise GGException("this function is called on single point")

    style = data.style or Style(
        line_width=1.0,
        color=BLACK,
        size=10.0,
    )
    ob = GOComposite(
        name=data.name if data.name else "error_bar",
        config=GraphicsObjectConfig(style=style),
        kind=CompositeKind.ERROR_BAR,
    )
    if ob.config.children is None:
        raise GGException("unexpected")

    if data.error_bar_kind == ErrorBarKind.LINES:
        create_lines_data = data.create_lines_input()
        res = create_lines(**create_lines_data)
        ob.config.children.append(res)

    elif data.error_bar_kind == ErrorBarKind.LINEST:
        if not style.size:
            raise GGException("expected style size")

        create_lines_data = data.create_lines_input()
        res = create_lines(**create_lines_data)
        ob.config.children.append(res)

        if data.axis_kind == AxisKind.X:
            scale2 = data.point.y.get_scale()
            q_ = Quantity.points(style.size)

            local_abs = q_.to_data(scale=scale2, length=data.view.point_height())

            low = coord_quantity_sub(data.point.y, local_abs)
            high = coord_quantity_add(data.point.y, local_abs)

            start = Coord(x=data.error_up, y=low)
            stop = Coord(x=data.error_up, y=high)

            right = init_line(
                start=start, stop=stop, init_line_input=InitLineInput(style=style)
            )

            start = Coord(x=data.error_down, y=low)
            stop = Coord(x=data.error_down, y=high)

            left = init_line(
                start=start, stop=stop, init_line_input=InitLineInput(style=style)
            )

            ob.config.children.extend([right, left])

        else:  # AxisKind.Y
            scale2 = data.point.x.get_scale()
            q_ = Quantity.points(style.size)

            local_abs = q_.to_data(scale=scale2, length=data.view.point_width())

            left_point = coord_quantity_sub(data.point.x, local_abs)
            right_point = coord_quantity_add(data.point.x, local_abs)

            start = Coord(x=left_point, y=data.error_up)
            stop = Coord(x=right_point, y=data.error_up)

            up = init_line(
                start=start, stop=stop, init_line_input=InitLineInput(style=style)
            )

            start = Coord(x=left_point, y=data.error_down)
            stop = Coord(x=right_point, y=data.error_down)

            down = init_line(
                start=start, stop=stop, init_line_input=InitLineInput(style=style)
            )

            ob.config.children.extend([up, down])

    return ob


def init_error_bar_for_point(data: InitErrorBarData) -> GraphicsObject:
    if not isinstance(data.point, Point):
        raise GGException("this function is called on multiple points")

    if data.view.x_scale is None or data.view.y_scale is None:
        raise GGException("view needs to have x and y scale")

    coord_data = deepcopy(data)
    coord_data.point = Coord(
        x=Coord1D.create_data(data.point.x, data.view.x_scale, AxisKind.X),
        y=Coord1D.create_data(data.point.y, data.view.y_scale, AxisKind.X),
    )
    return init_error_bar(coord_data)


def init_poly_line(
    pos: List[Coord],
    style: Optional[Style] = None,
    name: Optional[str] = None,
) -> GraphicsObject:
    style = style or Style(
        line_width=2.0,
        line_type=LineType.SOLID,
        color=BLACK,
        fill_color=TRANSPARENT,
    )

    return GOPolyLine(
        name=name or "polyline",
        config=GraphicsObjectConfig(style=style),
        pos=pos,
    )


def init_poly_line_from_points(
    view: ViewPort,
    pos: List[Point],
    style: Optional[Style] = None,
    name: Optional[str] = None,
) -> GraphicsObject:
    if view.x_scale is None or view.y_scale is None:
        raise GGException("expected x and y scale")

    positions = [
        Coord(
            x=Coord1D.create_data(p.x, view.x_scale, AxisKind.X),
            y=Coord1D.create_data(p.y, view.y_scale, AxisKind.Y),
        )
        for p in pos
    ]

    return init_poly_line(positions, style, name)


def _init_axis_label_data(axis_kind, view, margin_val, is_secondary, name):
    if axis_kind == AxisKind.X:
        y_pos = x_axis_y_pos(view=view, margin=margin_val, is_secondary=is_secondary)
        pos = Coord(x=Coord1D.create_relative(0.5), y=y_pos)
        name = f"y{name}"
        rotate_val = 0.0
        return pos, name, rotate_val
    else:  # AxisKind.Y
        x_pos = x_axis_y_pos(view=view, margin=margin_val, is_secondary=is_secondary)
        pos = Coord(x=x_pos, y=Coord1D.create_relative(0.5))
        name = f"y{name}"
        rotate_val = -90.0
        return pos, name, rotate_val


def init_axis_label(
    view: ViewPort,
    label: str,
    axis_kind: AxisKind,
    margin: Union[Quantity, Coord1D],
    font: Optional[Font] = None,
    name: Optional[str] = None,
    is_custom_margin: Optional[bool] = False,
    is_secondary: Optional[bool] = False,
    rotate: Optional[float] = None,
) -> GraphicsObject:
    name = name or ""

    margin_val = Quantity.centimeters(0.5).to_points().val if is_custom_margin else 0.0
    margin_min = Quantity.centimeters(1.0).to_points().val

    # todo improve this part
    if isinstance(margin, Quantity):
        margin_val += margin.to_points().val
    elif isinstance(margin, Coord1D):
        margin_val += margin.to_points().pos

    if margin_val < margin_min and not is_custom_margin:
        margin_val = margin_min

    pos, name, rotate_val = _init_axis_label_data(
        axis_kind, view, margin_val, is_secondary, name
    )

    if rotate is not None:
        rotate_val += rotate

    data = TextData(
        text=label,
        font=Font() if font is None else font,
        pos=pos,
        align=TextAlignKind.CENTER,
    )

    return GOText(
        name=name,
        config=GraphicsObjectConfig(rotate=rotate_val),
        data=data,
    )


def xlabel(
    view: ViewPort,
    label: str,
    margin: Coord1D,
    font: Optional[Font] = None,
    name: Optional[str] = None,
    is_secondary: Optional[bool] = None,
    rotate: Optional[float] = None,
) -> GraphicsObject:

    return init_axis_label(
        view=view,
        label=label,
        axis_kind=AxisKind.X,
        margin=margin,
        font=font,
        name=name,
        is_custom_margin=False,
        is_secondary=is_secondary,
        rotate=rotate,
    )


def ylabel(
    view: ViewPort,
    label: str,
    margin: Coord1D,
    font: Optional[Font] = None,
    name: Optional[str] = None,
    is_secondary: Optional[bool] = None,
    rotate: Optional[float] = None,
) -> GraphicsObject:

    return init_axis_label(
        view=view,
        label=label,
        axis_kind=AxisKind.Y,
        margin=margin,
        font=font,
        name=name,
        is_custom_margin=False,
        is_secondary=is_secondary,
        rotate=rotate,
    )


def xlabel_from_float(
    view: ViewPort,
    label: str,
    margin: float,
    font: Optional[Font] = None,
    name: Optional[str] = None,
    is_secondary: Optional[bool] = None,
    rotate: Optional[float] = None,
) -> GraphicsObject:

    return init_axis_label(
        view=view,
        label=label,
        axis_kind=AxisKind.X,
        margin=CentimeterUnit(margin),
        font=font,
        name=name,
        is_custom_margin=False,
        is_secondary=is_secondary,
        rotate=rotate,
    )


def ylabel_from_float(
    view: ViewPort,
    label: str,
    margin: float,
    font: Optional[Font] = None,
    name: Optional[str] = None,
    is_secondary: Optional[bool] = None,
    rotate: Optional[float] = None,
) -> GraphicsObject:

    return init_axis_label(
        view=view,
        label=label,
        axis_kind=AxisKind.Y,
        margin=CentimeterUnit(margin),
        font=font,
        name=name,
        is_custom_margin=False,
        is_secondary=is_secondary,
        rotate=rotate,
    )


# TODO start - this should move out
def x_label_origin_offset(
    view: ViewPort, font: Font, is_secondary: Optional[bool] = False
) -> Coord1D:
    pos = -1.15 if not is_secondary else 1.15
    return Coord1D.create_str_width(pos, font).to_relative(view.point_width())


def y_label_origin_offset(
    view: ViewPort, font: Font, is_secondary: Optional[bool] = False
) -> Coord1D:
    pos = 1.75 if not is_secondary else -1.75
    return Coord1D.create_str_height(pos, font).to_relative(view.point_height())


def set_text_align_kind(
    axis_kind: AxisKind,
    is_secondary: Optional[bool] = False,
    align_override: Optional[TextAlignKind] = None,
) -> TextAlignKind:
    if align_override is not None:
        return align_override

    if axis_kind == AxisKind.X:
        return TextAlignKind.CENTER
    else:  # axis_kind == AxisKind.Y
        return TextAlignKind.RIGHT if is_secondary else TextAlignKind.LEFT


# TODO end


def init_tick_label_with_override(
    view: ViewPort,
    tick: GOTick,
    label_text: str,
    axis_kind: AxisKind,
    data: InitTextInput,
    align_override: Optional[TextAlignKind] = None,
    is_secondary: Optional[bool] = None,
    margin: Optional[Coord1D] = None,
) -> GOText:

    font_ = data.font or Font(size=1.0)
    loc = tick.pos
    align_to = set_text_align_kind(axis_kind, is_secondary, align_override)

    data = InitTextInput(
        text=label_text,
        align_kind=align_to,
        font=font_,
        rotate=data.rotate,
        name=data.name,
    )

    if tick.axis == AxisKind.X:
        y_offset = margin or y_label_origin_offset(view, font_, is_secondary)
        origin = Coord(x=loc.x, y=(loc.y + y_offset).to_relative(None))
        return init_text(view, origin, data)

    elif tick.axis == AxisKind.Y:
        x_offset = margin or y_label_origin_offset(view, font_, is_secondary)
        origin = Coord(x=(loc.x + x_offset).to_relative(None), y=loc.y)
        return init_text(view, origin, data)
    else:
        raise GGException("unexpected axis")


def axis_coord(
    coord: Coord1D, axis_kind: AxisKind, is_secondary: Optional[bool] = None
) -> Coord:
    if axis_kind == AxisKind.X:
        return Coord(x=coord, y=x_axis_y_pos(None, None, is_secondary))
    else:  # AxisKind.Y
        return Coord(x=y_axis_x_pos(None, None, is_secondary), y=coord)


TickFormat = Callable[[float], str]


@dataclass
class TickLabelsInput:
    font: Optional["Font"] = None
    is_secondary: Optional[bool] = None
    margin: Optional[Coord1D] = None
    format_fn: Optional[TickFormat] = None
    rotate: Optional[float] = None
    align_to_override: Optional["TextAlignKind"] = None


def tick_labels(
    view: ViewPort, ticks: List[GOTick], tick_labels_input: TickLabelsInput
) -> List[GraphicsObject]:
    # todo move to view ?
    if not ticks:
        raise GGException("empty ticks vector")

    axis_kind = ticks[0].axis
    tick_labels_input.font = tick_labels_input.font or Font(size=12.0)

    positions = [x.pos.x.pos if axis_kind == AxisKind.X else x.pos.y.pos for x in ticks]

    max_pos = max(positions)
    min_pos = min(positions)

    tick_scale = (max_pos - min_pos) / (len(positions) - 1)

    def default_format(f: float) -> str:
        return format_tick_value(f, tick_scale)

    format_fn = tick_labels_input.format_fn or default_format

    strs = [format_fn(p) for p in positions]
    strs_len = len(strs)
    strs_unique = list(dict.fromkeys(strs))  # Unique while preserving order
    strs_unique_len = len(strs_unique)

    rotate = tick_labels_input.rotate
    new_positions = []
    result: List[GraphicsObject] = []

    if strs_unique_len < strs_len:
        new_positions = [x - min_pos for x in positions]
        max_tick = ticks[-1]

        if axis_kind == AxisKind.X:
            coord = axis_coord(
                max_tick.pos.x, AxisKind.X, tick_labels_input.is_secondary
            )
            pos = (
                coord.y.to_points(view.h_img).pos
                - Quantity.centimeters(1.5).to_points().val
            )
            coord.y = Coord1D.create_point(pos)
        else:  # AxisKind.Y
            coord = axis_coord(
                max_tick.pos.y, AxisKind.Y, tick_labels_input.is_secondary
            )
            pos = (
                coord.x.to_points(view.w_img).pos
                - Quantity.centimeters(2.0).to_points().val
            )
            rotate = rotate or -90.0
            coord.y = Coord1D.create_point(pos, None)

        text = f"+{format_fn(min_pos)}"
        data = InitTextInput(
            text=text,
            align_kind=TextAlignKind.RIGHT,
            font=tick_labels_input.font,
            rotate=rotate,
            name="axis_substraction",
        )
        new_text = init_text(view, coord, data)
        result.append(new_text)

    for idx, obj in enumerate(ticks):
        label_text = format_fn(new_positions[idx]) if new_positions else strs[idx]

        data = InitTextInput(
            align_kind=TextAlignKind.RIGHT,
            font=tick_labels_input.font,
            rotate=rotate,
            name="tickLabel",
        )
        new_tick_label = init_tick_label_with_override(
            view=view,
            tick=obj,
            label_text=label_text,
            axis_kind=axis_kind,
            data=data,
        )
        result.append(new_tick_label)

    return result


def init_tick(
    view: ViewPort,
    axis_kind: AxisKind,
    major: bool,
    at: Coord,
    tick_kind: Optional[TickKind] = None,
    style: Optional[Style] = None,
    name: Optional[str] = None,
    is_secondary: Optional[bool] = False,
) -> GOTick:
    name = name or "tick"
    tick_kind = tick_kind or TickKind.ONE_SIDE

    default_style = Style(
        line_width=1.0, color=BLACK, size=5.0, line_type=LineType.SOLID
    )
    style = style or default_style

    return GOTick(
        name=name,
        config=GraphicsObjectConfig(style=style),
        major=major,
        pos=path_coord_view_port(at, view),
        axis=axis_kind,
        kind=tick_kind,
        secondary=is_secondary,
    )


def tick_labels_from_coord(
    view: "ViewPort",
    tick_pos: list["Coord1D"],
    tick_labels_list: list[str],
    axis_kind: AxisKind,
    font: Optional[Font] = None,
    is_secondary: Optional[bool] = False,
    rotate: Optional[float] = None,
    margin: Optional[Coord1D] = None,
    align_override: Optional[TextAlignKind] = None,
) -> list[tuple[GOTick, GOText]]:

    if len(tick_pos) != len(tick_labels_list):
        raise GGException("Must have as many tick positions as labels")

    font = font or Font(size=8.0)

    result = []
    for idx, pos in enumerate(tick_pos):
        at = axis_coord(pos, axis_kind, is_secondary)
        tick = init_tick(
            view=view,
            axis_kind=axis_kind,
            major=True,
            at=at,
            tick_kind=None,
            style=None,
            name=None,
            is_secondary=is_secondary,
        )

        data = InitTextInput(font=font, rotate=rotate)

        result.append(
            (
                tick,
                init_tick_label_with_override(
                    view=view,
                    tick=tick,
                    label_text=tick_labels_list[idx],
                    axis_kind=axis_kind,
                    data=data,
                    align_override=align_override,
                    is_secondary=is_secondary,
                    margin=margin,
                ),
            )
        )

    return result


def calc_tick_locations(scale: Scale, num_ticks: int) -> Tuple[Scale, float, int]:
    if scale.low == scale.high:
        raise GGException("a data scale is required to calculate tick positions")

    axis_end = scale.high
    axis_start = scale.low
    # axis_width = axis_end - axis_start

    nice_range = nice_number(axis_end, False)
    nice_tick = nice_number(nice_range / (num_ticks - 1), True)

    new_axis_start = math.floor(axis_start / nice_tick) * nice_tick
    new_axis_end = math.ceil(axis_end / nice_tick) * nice_tick

    new_scale = Scale(low=new_axis_start, high=new_axis_end)
    num_ticks_actual = round((new_axis_end - new_axis_start) / nice_tick)

    return new_scale, nice_tick, num_ticks_actual


def filter_by_bound_scale(
    tick_pos: List[Coord], axis_kind: AxisKind, bound_scale: Optional[Scale] = None
) -> List[Coord]:
    if bound_scale is not None:
        result = [
            coord
            for coord in tick_pos
            if bound_scale.low
            <= coord.dimension_for_axis(axis_kind).pos
            <= bound_scale.high
        ]
        return result
    else:
        return tick_pos


def init_ticks(
    view: ViewPort,
    axis_kind: AxisKind,
    tick_locs: List[Coord],
    num_ticks: Optional[int] = None,
    tick_kind: Optional[TickKind] = None,
    major: bool = True,
    style: Optional[Style] = None,
    update_scale: bool = True,
    is_secondary: Optional[bool] = None,
    bound_scale: Optional[Scale] = None,
) -> List[GraphicsObject]:
    result: List[GraphicsObject] = []
    num_ticks = num_ticks or 0
    tick_kind = tick_kind or TickKind.ONE_SIDE

    if num_ticks == 0 and not tick_locs:
        raise GGException("need to provide num_ticks or tick_locks")

    if num_ticks == 0 and tick_locs:
        for location in tick_locs:
            new_obj = init_tick(
                view,
                axis_kind,
                major,
                location,
                tick_kind=tick_kind,
                style=style,
                is_secondary=is_secondary,
            )
            result.append(new_obj)
    elif num_ticks > 0:
        scale = view.scale_for_axis(axis_kind)
        if scale is None:
            raise GGException("expected scales on view")

        new_scale, _, new_num_ticks = calc_tick_locations(scale, num_ticks)
        tick_scale = bound_scale or new_scale

        temp = linspace(new_scale.low, new_scale.high, new_num_ticks + 1)
        auto_tick_locations = [
            axis_coord(
                Coord1D.create_data(pos, tick_scale, axis_kind),
                axis_kind,
                is_secondary,
            )
            for pos in temp
        ]

        auto_tick_locations = filter_by_bound_scale(
            auto_tick_locations, axis_kind, bound_scale
        )
        result = init_ticks(
            view,
            axis_kind,
            num_ticks=None,
            tick_locs=auto_tick_locations,
            tick_kind=tick_kind,
            major=major,
            style=style,
            update_scale=update_scale,
            is_secondary=is_secondary,
            bound_scale=bound_scale,
        )

        if axis_kind == AxisKind.X:
            view.x_scale = tick_scale
        else:
            view.y_scale = tick_scale

    if update_scale:
        view.update_data_scale()

    return result


def xticks(
    view: ViewPort,
    tick_locs: List[Coord],
    num_ticks: Optional[int] = None,
    tick_kind: Optional[TickKind] = None,
    style: Optional[Style] = None,
    update_scale: bool = True,
    is_secondary: bool = False,
    bound_scale: Optional[Scale] = None,
) -> List[GraphicsObject]:
    return init_ticks(
        view,
        AxisKind.X,
        tick_locs,
        num_ticks=num_ticks or 10,
        tick_kind=tick_kind,
        major=True,
        style=style,
        update_scale=update_scale,
        is_secondary=is_secondary,
        bound_scale=bound_scale,
    )


def yticks(
    view: ViewPort,
    tick_locs: List[Coord],
    num_ticks: Optional[int] = None,
    tick_kind: Optional[TickKind] = None,
    style: Optional[Style] = None,
    update_scale: bool = True,
    is_secondary: bool = False,
    bound_scale: Optional[Scale] = None,
) -> List[GraphicsObject]:
    return init_ticks(
        view,
        AxisKind.Y,
        tick_locs,
        num_ticks=num_ticks or 10,
        tick_kind=tick_kind,
        major=True,
        style=style,
        update_scale=update_scale,
        is_secondary=is_secondary,
        bound_scale=bound_scale,
    )


def calc_minor_ticks(ticks: List[GOTick], axis_kind: AxisKind) -> List[Coord1D]:
    result = []
    first = ticks[0]

    scale = first.scale_for_axis(axis_kind)

    if scale is None:
        raise GGException("scale not found")

    cdiv2 = Coord1D.create_data(2.0, scale=scale, axis_kind=axis_kind)

    for i, tick in enumerate(ticks):
        if axis_kind == AxisKind.X:
            result.append((tick.pos.x + ticks[i + 1].pos.x) / cdiv2)
        else:  # AxisKind.Y
            result.append((tick.pos.y + ticks[i + 1].pos.y) / cdiv2)

    return result


def init_grid_lines(
    x_ticks=None, y_ticks=None, major: bool = True, style=None, name="grid_lines"
) -> GOGrid:
    default_style = Style(
        size=1.0 if major else 0.3, color=WHITE, line_type=LineType.SOLID
    )

    style = style or default_style
    x_ticks = x_ticks or []
    y_ticks = y_ticks or []

    if major:
        x_ticks = [obj.get_pos().x for obj in x_ticks]
        y_ticks = [obj.get_pos().y for obj in y_ticks]
    else:
        x_ticks = calc_minor_ticks(x_ticks, AxisKind.X)
        y_ticks = calc_minor_ticks(y_ticks, AxisKind.Y)

    return GOGrid(
        name=name,
        config=GraphicsObjectConfig(style=style),
        x_pos=x_ticks,
        y_pos=y_ticks,
    )