from math import pi

import cairo
from cairo import (
    FONT_SLANT_ITALIC,
    FONT_SLANT_NORMAL,
    FONT_SLANT_OBLIQUE,
    FONT_WEIGHT_BOLD,
    FORMAT_ARGB32,
    Context,
    FontWeight,
    Gradient,
    ImageSurface,
    LinearGradient,
    TextExtents,
)


def to_cairo_font_slaint(c_font_slant):
    from python_ggplot.core.objects import CFontSlant

    data = {
        CFontSlant.NORMAL: FONT_SLANT_NORMAL,
        CFontSlant.ITALIC: FONT_SLANT_ITALIC,
        CFontSlant.OBLIQUE: FONT_SLANT_OBLIQUE,
    }
    return data[c_font_slant]


def to_cairo_font_weight(font):
    data = {True: FontWeight.NORMAL, False: FontWeight.BOLD}
    return data[font.bold]


def create_gradient(
    gradient: Gradient, left: float, bottom: float, height: float, width: float
) -> LinearGradient:
    # todo revisit this later
    # middle = bottom + height / 2.0
    # right = left + width
    center = left + width / 2.0

    result = LinearGradient(center, bottom + height, center, bottom)

    # step_size = width / len(gradient.colors)
    num_colors = len(gradient.colors)

    for i, color in enumerate(gradient.colors):
        result.add_color_stop_rgb(i / num_colors, color.r, color.g, color.b)

    return result


class CairoBackend:
    def __init__(self, canvas):
        self.canvas = canvas
        self.ctx = None
        self.created = False
        self.text_extents = None
        # if surface:
        #     self.ctx = Context(surface)
        #     self.created = True

    @staticmethod
    def with_surface(img, cb):
        # todo refacor
        if not img.backend.created:
            # surface = ImageSurface(FORMAT_ARGB32, img.width, img.height)
            img.backend.ctx = Context(img.backend.canvas)
            # self.canvas = surface
            img.backend.created = True
        if img.backend.ctx:
            img.backend.ctx.save()
            cb(img.backend.ctx)
            img.backend.ctx.restore()

    def rotate(self, ctx, angle, around):
        ctx.translate(around.x, around.y)
        ctx.rotate(angle * pi / 180.0)
        ctx.translate(-around.x, -around.y)

    def get_line_style(self, line_type, line_width):
        from python_ggplot.core.objects import LineType

        dash = line_width * 4.0
        dash_space = line_width * 5.0
        dot = line_width / 2.0
        dot_space = line_width * 2.0
        long_dash = line_width * 8.0

        lookup = {
            LineType.DASHED: [dash, dash_space],
            LineType.DOTTED: [dot, dot_space],
            LineType.DOT_DASH: [dot, dot_space, dash, dot_space],
            LineType.LONG_DASH: [long_dash, dash_space],
            LineType.TWO_DASH: [dash, dot_space * 2.0, long_dash, dot_space * 2.0],
        }
        return lookup.get(line_type, [])

    def set_line_style(self, ctx, line_type, line_width):
        from python_ggplot.core.objects import LineType

        if line_type == LineType.NONE_TYPE:
            ctx.set_line_width(0.0)
        else:
            style = self.get_line_style(line_type, line_width)
            ctx.set_dash(style, len(style))

    def draw_line(self, img, start, stop, style, rotate_angle=None):
        def callback(context):
            if rotate_angle:
                self.rotate(context, rotate_angle[0], rotate_angle[1])

            context.set_source_rgba(
                style.color.r,
                style.color.g,
                style.color.b,
                style.color.a,
            )
            self.set_line_style(context, style.line_type, style.line_width)
            context.set_line_width(style.line_width)
            context.move_to(start.x, start.y)
            context.line_to(stop.x, stop.y)
            context.stroke()

        CairoBackend.with_surface(img, callback)

    def draw_polyline(self, img, points, style, rotate_angle=None):
        def callback(context):
            if rotate_angle:
                self.rotate(context, rotate_angle[0], rotate_angle[1])
            context.set_source_rgba(
                style.color.r,
                style.color.g,
                style.color.b,
                style.color.a,
            )
            self.set_line_style(context, style.line_type, style.line_width)
            context.set_line_width(style.line_width)

            context.move_to(points[0].x, points[0].y)
            for point in points[1:]:
                context.line_to(point.x, point.y)
            context.stroke_preserve()
            context.set_source_rgba(
                style.fill_color.r,
                style.fill_color.g,
                style.fill_color.b,
                style.fill_color.a,
            )
            context.fill()

        CairoBackend.with_surface(img, callback)

    def draw_circle(
        self,
        img,
        center,
        radius,
        line_width,
        stroke_color=None,
        fill_color=None,
        rotate_angle=None,
    ):
        if not fill_color:
            fill_color = Color(r=0.0, g=0.0, b=0.0, a=0.0)

        if not stroke_color:
            stroke_color = Color(r=0.0, g=0.0, b=0.0, a=0.0)

        def callback(context):
            if rotate_angle:
                self.rotate(context, rotate_angle[0], rotate_angle[1])

            context.set_line_width(line_width)
            context.set_source_rgba(
                stroke_color.r, stroke_color.g, stroke_color.b, stroke_color.a
            )
            context.arc(center[0], center[1], radius, 0.0, 2.0 * pi)
            context.stroke_preserve()
            context.set_source_rgba(
                fill_color.r, fill_color.g, fill_color.b, fill_color.a
            )
            context.fill()

        CairoBackend.with_surface(img, callback)

    def rotate_if_needed(self, context, rotate_angle):
        if rotate_angle:
            self.rotate(context, rotate_angle[0], rotate_angle[1])

    @classmethod
    def get_text_extends_from_context(cls, context, text) -> TextExtents:
        return context.text_extents(text)

    @classmethod
    def get_text_extend(cls, text, font) -> TextExtents:
        width = len(text) * font.size * 2.0
        height = font.size * 2.0
        surface = ImageSurface(FORMAT_ARGB32, int(width), int(height))
        context = Context(surface)

        context.select_font_face(
            font.family, to_cairo_font_slaint(font.slant), to_cairo_font_weight(font)
        )
        context.set_font_size(font.size)
        context.set_source_rgba(font.color.r, font.color.g, font.color.b, font.color.a)
        result = cls.get_text_extends_from_context(context, text)
        return result

    def draw_text(
        self, img, text, font, at, align_kind=None, rotate=None, rotate_in_view=None
    ):
        from python_ggplot.core.objects import GGException, TextAlignKind

        if align_kind is None:
            align_kind = TextAlignKind.LEFT

        def callback(context):
            if rotate_in_view:
                self.rotate(context, rotate_in_view[0], rotate_in_view[1])
            context.select_font_face(font.family, FONT_SLANT_NORMAL, FONT_WEIGHT_BOLD)
            context.set_font_size(font.size)
            context.set_source_rgba(
                font.color.r, font.color.g, font.color.b, font.color.a
            )
            x, y = at.x, at.y
            extends = CairoBackend.get_text_extends_from_context(context, text)

            if rotate:
                self.rotate(context, rotate, at)

            if align_kind == TextAlignKind.LEFT:
                move_to_x = x
                move_to_y = y - extends.height / 2.0 + extends.y_bearing
            elif align_kind == TextAlignKind.RIGHT:
                move_to_x = x - (extends.width + extends.x_bearing)
                move_to_y = y - (extends.height / 2.0 + extends.y_bearing)
            elif align_kind == TextAlignKind.CENTER:  # Center
                move_to_x = x - (extends.width / 2.0 + extends.y_bearing)
                move_to_y = y - (extends.height / 2.0 + extends.y_bearing)
            else:
                raise GGException("unexpected")

            context.move_to(move_to_x, move_to_y)
            context.show_text(text)

        CairoBackend.with_surface(img, callback)

    def draw_rectangle(
        self, img, left, bottom, width, height, style, rotate=None, rotate_in_view=None
    ):
        def callback(context):
            if rotate_in_view:
                self.rotate(context, rotate_in_view[0], rotate_in_view[1])
            if rotate:
                self.rotate(context, rotate, [left, bottom])
            context.rectangle(left, bottom, width, height)
            context.set_line_width(style.line_width)
            self.set_line_style(context, style.line_type, style.line_width)
            context.set_source_rgba(
                style.color.r,
                style.color.g,
                style.color.b,
                style.color.a,
            )
            context.stroke_preserve()

            if style.gradient:
                pattern = create_gradient(style.gradient, left, bottom, height, width)
                context.set_source(pattern)
            else:
                context.set_source_rgb(
                    style.fill_color.r,
                    style.fill_color.g,
                    style.fill_color.b,
                )
            context.fill()

        CairoBackend.with_surface(img, callback)

    def draw_raster(
        self,
        img,
        left,
        bottom,
        width,
        height,
        num_x,
        num_y,
        draw_cb,
        rotate=None,
        rotate_in_view=None,
    ):
        def callback(context):
            if rotate_in_view:
                self.rotate(context, rotate_in_view[0], rotate_in_view[1])
            if rotate:
                self.rotate(context, rotate, [left, bottom])

            w_img = int(width)
            h_img = int(height)

            to_draw = draw_cb()

            block_size_x = width / num_x
            block_size_y = height / num_y

            data = [0] * (w_img * h_img)
            for y in range(h_img):
                for x in range(w_img):
                    tx = int(x / block_size_x)
                    ty = int(y / block_size_y)
                    data_idx = y * w_img + x
                    to_draw_val = ty * num_x + tx
                    data[data_idx] = to_draw[to_draw_val]

            data2 = self.to_bytes(data)
            stride = w_img * 4
            surface = ImageSurface.create_for_data(
                data2, FORMAT_ARGB32, w_img, h_img, stride
            )
            surface.mark_dirty()
            context.set_source_surface(surface, left, bottom)
            context.paint()
            del surface

        CairoBackend.with_surface(img, callback)

    def to_bytes(self, input_data):
        return bytearray(input_data)


def init_image(filename, width, height, ftype):
    # todo fix this later
    from python_ggplot.core.objects import FileTypeKind, GGException, Image

    if ftype == FileTypeKind.PNG:
        surface = ImageSurface(FORMAT_ARGB32, width, height)
        backend = CairoBackend(surface)
        return Image(filename, width, height, ftype, backend)
    else:
        raise GGException("Only PNG format is supported for now")


def cairo_test():
    surface = ImageSurface(FORMAT_ARGB32, 600, 400)
    context = Context(surface)

    context.select_font_face("serif", FONT_SLANT_NORMAL, FONT_WEIGHT_BOLD)
    context.set_font_size(32.0)
    context.set_source_rgb(0.0, 0.0, 1.0)
    context.move_to(100.0, 300.0)
    context.show_text("Hello from Python!")

    with open("cairo_test.png", "wb") as buffer:
        surface.write_to_png(buffer)
