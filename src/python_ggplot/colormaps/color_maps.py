import json
import os


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

color_maps_data = ColorMapsData()
# TODO Do we really want to load a json file at import time?
# This can be changed if we decide to
# i dont like reading it at import time, but the json file small
# revisit down the line
VIRIDIS_RAW = color_maps_data.viridis_raw
MAGMARAW = color_maps_data.magmaraw
INFERNO_RAW = color_maps_data.inferno_raw
PLASMA_RAW = color_maps_data.plasma_raw
