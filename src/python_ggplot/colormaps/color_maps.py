import json
import os
from typing import List

from python_ggplot.core.objects import ColorRGBA
from python_ggplot.gg_scales import ColorScale


class ColorMapsData:
    def __init__(self):
        self._data = None

    @property
    def data(self):
        if self._data is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "maps.json")

            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                self._data = data
        return self._data

    @property
    def viridis_raw(self):
        return self.data["viridis_raw"]

    @property
    def magmaraw(self):
        return self.data["magmaraw"]

    @property
    def inferno_raw(self):
        return self.data["inferno_raw"]

    @property
    def plasma_raw(self):
        return self.data["plasma_raw"]


def _to_val(x: float):
    # TODO, add tests for this and make sure
    # it only happens at the right times
    # i dont like silencing errors by shifting the value to valid bounds
    # if it silnences things that are actual errors
    # and not out of bounds because expected, this can be very hard to debug
    return max(0, min(int(round(x * 255.0)), 255))


def to_color_scale(name: str, color_map: List[List[float]]):
    colors: List[int] = []
    for r, g, b in color_map:
        new_col = (255 << 24) | (_to_val(r) << 16) | (_to_val(g) << 8) | (_to_val(b))
        colors.append(new_col)
    result = ColorScale(name=name, colors=colors)
    return result


def to_rgba(c: int) -> tuple[int, int, int, int]:
    # todo decide what stays in color maps and what goes in chroma
    return (
        (c >> 16) & 0xFF,  # red
        (c >> 8) & 0xFF,  # green
        c & 0xFF,  # blue
        (c >> 24) & 0xFF,  # alpha
    )


def int_to_color(c: int) -> ColorRGBA:
    r, g, b, a = to_rgba(c)
    return ColorRGBA(r=r, g=g, b=b, a=a)


def parse_html_color(c: str):
    # port from chroma
    raise NotImplemented()


def value_to_color(c: int | str) -> ColorRGBA:
    # TODO this is mean to use Value class
    # port bit by bit
    if isinstance(c, int):
        return int_to_color(c)
    if isinstance(c, str):
        return parse_html_color(c)
    raise ValueError("expected str or int")


color_maps_data = ColorMapsData()
# TODO Do we really want to load a json file at import time?
# This can be changed if we decide to
# i dont like reading it at import time, but the json file small
# revisit down the line
VIRIDIS_RAW = color_maps_data.viridis_raw
MAGMARAW = color_maps_data.magmaraw
INFERNO_RAW = color_maps_data.inferno_raw
PLASMA_RAW = color_maps_data.plasma_raw


VIRIDIS_RAW_COLOR_SCALE = to_color_scale("viridis", VIRIDIS_RAW)
MAGMARAW_COLOR_SCALE = to_color_scale("magma", MAGMARAW)
INFERNO_RAW_COLOR_SCALE = to_color_scale("inferno", INFERNO_RAW)
PLASMA_RAW_COLOR_SCALE = to_color_scale("plasma", PLASMA_RAW)
