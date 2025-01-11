import cairo
from math import pi


class CairoBackend:
    def __init__(self, canvas):
        self.canvas = canvas
        self.ctx = None
        self.created = False
        self.text_extents = None
        if surface:
            self.ctx = cairo.Context(surface)
            self.created = True

    def with_surface(self, img, cb):
        if not self.created:
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img.width, img.height)
            self.ctx = cairo.Context(surface)
            self.canvas = surface
            self.created = True
        if self.ctx:
            self.ctx.save()
            cb(self.ctx)
            self.ctx.restore()
    
    def rotate(self, ctx, angle, around):
        ctx.translate(around[0], around[1])
        ctx.rotate(angle * pi / 180.0)
        ctx.translate(-around[0], -around[1])
    
    def get_line_style(self, line_type, line_width):
        dash = line_width * 4.0
        dash_space = line_width * 5.0
        dot = line_width / 2.0
        dot_space = line_width * 2.0
        long_dash = line_width * 8.0
        
        if line_type == 'Dashed':
            return [dash, dash_space]
        elif line_type == 'Dotted':
            return [dot, dot_space]
        elif line_type == 'DotDash':
            return [dot, dot_space, dash, dot_space]
        elif line_type == 'LongDash':
            return [long_dash, dash_space]
        elif line_type == 'TwoDash':
            return [dash, dot_space * 2.0, long_dash, dot_space * 2.0]
        else:
            return []
    
    def set_line_style(self, ctx, line_type, line_width):
        if line_type == 'None':
            ctx.set_line_width(0.0)
        else:
            style = self.get_line_style(line_type, line_width)
            ctx.set_dash(style, len(style))
    
    def draw_line(self, img, start, stop, style, rotate_angle=None):
        def callback(context):
            if rotate_angle:
                self.rotate(context, rotate_angle[0], rotate_angle[1])
            context.set_source_rgba(style['color'][0], style['color'][1], style['color'][2], style['color'][3])
            self.set_line_style(context, style['line_type'], style['line_width'])
            context.set_line_width(style['line_width'])
            context.move_to(start[0], start[1])
            context.line_to(stop[0], stop[1])
            context.stroke()
        
        self.with_surface(img, callback)
    
    def draw_polyline(self, img, points, style, rotate_angle=None):
        def callback(context):
            if rotate_angle:
                self.rotate(context, rotate_angle[0], rotate_angle[1])
            context.set_source_rgba(style['color'][0], style['color'][1], style['color'][2], style['color'][3])
            self.set_line_style(context, style['line_type'], style['line_width'])
            context.set_line_width(style['line_width'])
            
            context.move_to(points[0][0], points[0][1])
            for point in points[1:]:
                context.line_to(point[0], point[1])
            context.stroke_preserve()
            context.set_source_rgba(style['fill_color'][0], style['fill_color'][1], style['fill_color'][2], style['fill_color'][3])
            context.fill()
        
        self.with_surface(img, callback)
    
    def draw_circle(self, img, center, radius, line_width, stroke_color=None, fill_color=None, rotate_angle=None):
        if not fill_color:
            fill_color = [0.0, 0.0, 0.0, 0.0]
        if not stroke_color:
            stroke_color = [0.0, 0.0, 0.0, 0.0]
        
        def callback(context):
            if rotate_angle:
                self.rotate(context, rotate_angle[0], rotate_angle[1])
            context.set_line_width(line_width)
            context.set_source_rgba(stroke_color[0], stroke_color[1], stroke_color[2], stroke_color[3])
            context.arc(center[0], center[1], radius, 0.0, 2.0 * pi)
            context.stroke_preserve()
            context.set_source_rgba(fill_color[0], fill_color[1], fill_color[2], fill_color[3])
            context.fill()
        
        self.with_surface(img, callback)
    
    def rotate_if_needed(self, context, rotate_angle):
        if rotate_angle:
            self.rotate(context, rotate_angle[0], rotate_angle[1])
    
    def get_text_extents_from_context(self, context, text):
        return context.text_extents(text)
    
    def get_text_extents(self, text, font):
        width = len(text) * font['size'] * 2.0
        height = font['size'] * 2.0
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width), int(height))
        context = cairo.Context(surface)

        context.select_font_face(font['family'], cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(font['size'])
        context.set_source_rgba(font['color'][0], font['color'][1], font['color'][2], font['color'][3])
        result = self.get_text_extents_from_context(context, text)
        return result
    
    def draw_text(self, img, text, font, at, align_kind=None, rotate=None, rotate_in_view=None):
        if align_kind is None:
            align_kind = 'Left'
        
        def callback(context):
            if rotate_in_view:
                self.rotate(context, rotate_in_view[0], rotate_in_view[1])
            context.select_font_face(font['family'], cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            context.set_font_size(font['size'])
            context.set_source_rgba(font['color'][0], font['color'][1], font['color'][2], font['color'][3])
            x, y = at
            extends = self.get_text_extents_from_context(context, text)
            
            if rotate:
                self.rotate(context, rotate, at)
            
            if align_kind == 'Left':
                move_to_x = x
                move_to_y = y - extends.height / 2.0 + extends.y_bearing
            elif align_kind == 'Right':
                move_to_x = x - (extends.width + extends.x_bearing)
                move_to_y = y - (extends.height / 2.0 + extends.y_bearing)
            else:  # Center
                move_to_x = x - (extends.width / 2.0 + extends.y_bearing)
                move_to_y = y - (extends.height / 2.0 + extends.y_bearing)
            
            context.move_to(move_to_x, move_to_y)
            context.show_text(text)
        
        self.with_surface(img, callback)
    
    def draw_rectangle(self, img, left, bottom, width, height, style, rotate=None, rotate_in_view=None):
        def callback(context):
            if rotate_in_view:
                self.rotate(context, rotate_in_view[0], rotate_in_view[1])
            if rotate:
                self.rotate(context, rotate, [left, bottom])
            context.rectangle(left, bottom, width, height)
            context.set_line_width(style['line_width'])
            self.set_line_style(context, style['line_type'], style['line_width'])
            context.set_source_rgba(style['color'][0], style['color'][1], style['color'][2], style['color'][3])
            context.stroke_preserve()
            
            if style.get('gradient'):
                pattern = self.create_gradient(style['gradient'], left, bottom, height, width)
                context.set_source(pattern)
            else:
                context.set_source_rgb(style['fill_color'][0], style['fill_color'][1], style['fill_color'][2])
            context.fill()
        
        self.with_surface(img, callback)

    def draw_raster(self, img, left, bottom, width, height, num_x, num_y, draw_cb, rotate=None, rotate_in_view=None):
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
                    to_draw_val = (ty * num_x + tx)
                    data[data_idx] = to_draw[to_draw_val]
            
            data2 = self.to_bytes(data)
            surface = cairo.ImageSurface.create_for_data(data2, cairo.FORMAT_ARGB32, w_img, h_img)
            surface.mark_dirty()
            context.set_source_surface(surface, left, bottom)
            context.paint()
        
        self.with_surface(img, callback)
    
    def to_bytes(self, input_data):
        return bytearray(input_data)

def init_image(backend, filename, width, height, ftype):
    if ftype == 'Png':
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    else:
        raise ValueError("Only PNG format is supported")

    backend = CairoBackend(surface)
    return Image(filename, width, height, ftype, backend)


def create_gradient(gradient, left, bottom, height, width):
    middle = bottom + height / 2.0
    right = left + width
    center = left + width / 2.0
    result = cairo.LinearGradient(center, bottom + height, center, bottom)
    
    step_size = width / len(gradient['colors'])
    num_colors = len(gradient['colors'])

    for i, color in enumerate(gradient['colors']):
        result.add_color_stop_rgb(i / num_colors, color['r'], color['g'], color['b'])
    
    return result

def cairo_test():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 600, 400)
    context = cairo.Context(surface)

    context.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(32.0)
    context.set_source_rgb(0.0, 0.0, 1.0)
    context.move_to(100.0, 300.0)
    context.show_text("Hello from Python!")

    with open("foo.png", "wb") as buffer:
        surface.write_to_png(buffer)

