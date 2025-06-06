from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import pandas as pd

from python_ggplot.core.objects import GGException, Scale
from python_ggplot.gg.geom.base import Geom, GeomType

if TYPE_CHECKING:
    from python_ggplot.gg.scales.base import ColorScale, FilledScales


def _get_width(
    geom: Geom, fs: "FilledScales", df: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    # todo clean up
    width_scale = fs.get_width_scale(geom, optional=True)
    x_min_s = fs.get_x_min_scale(geom, optional=True)
    x_max_s = fs.get_x_max_scale(geom, optional=True)
    # Handle width
    if width_scale is not None:
        width = width_scale.get_col_name()  # type: ignore
        return width, df

    elif x_min_s is not None and x_max_s is not None:
        min_name = x_min_s.get_col_name()  # type: ignore
        max_name = x_max_s.get_col_name()  # type: ignore
        x_col_name = get_x_scale(fs, fg.geom).get_col_name()  # type: ignore
        df["width"] = df[max_name] - df[min_name]
        df[x_col_name] = df[min_name]
        return "width", df
    elif x_min_s is not None or x_max_s is not None:
        raise GGException(
            "Invalid combination of aesthetics! If no width given both an `x_min` and `x_max` has to be supplied for geom_{fg.geom_kind}!"
        )
    else:
        if geom.geom_type == GeomType.RASTER:
            x_col = df[get_x_scale(fs, fg.geom).get_col_name()].unique()  # type: ignore
            fg.num_x = len(x_col)  # type: ignore
            df["width"] = abs(x_col[1] - x_col[0])  # type: ignore
        else:
            print(
                "INFO: using default width of 1 since no width information supplied. "
                "Add `width` or (`x_min`, `x_max`) as aesthetics for different values."
            )
            df["width"] = 1.0
        return "width", df


def _get_height(
    geom: Geom, fs: "FilledScales", df: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    # todo clean up
    height_scale = fs.get_height_scale(geom, optional=True)
    y_min_s = fs.get_y_min_scale(geom, optional=True)
    y_max_s = fs.get_y_max_scale(geom, optional=True)

    if height_scale is not None:
        return height_scale.get_col_name(), df

    elif y_min_s is not None and y_max_s is not None:
        min_name = y_min_s.get_col_name()
        max_name = y_max_s.get_col_name()

        y_scale = fs.get_y_scale(geom)
        if y_scale is None:
            raise GGException("expected a y_scale")

        y_col_name = y_scale.get_col_name()
        df["height"] = df[max_name] - df[min_name]
        df[y_col_name] = df[min_name]
        return "height", df

    elif y_min_s is not None or y_max_s is not None:
        raise GGException(
            "Invalid combination of aesthetics! If no height given both an `y_min` and `y_max` has to be supplied for geom_{fg.geom_kind}!"
        )
    else:
        if geom.geom_type == GeomType.RASTER:
            col_name = get_y_scale(fs, fg.geom).get_col_name()  # type: ignore
            y_col = df[col_name].unique()  # type: ignore
            fg.num_y = len(y_col)  # type: ignore
            df["height"] = abs(y_col[1] - y_col[0])  # type: ignore
        else:
            print(
                "INFO: using default height of 1 since no height information supplied. "
                "Add `height` or (`y_min`, `y_max`) as aesthetics for different values."
            )
            df["height"] = 1.0
        return "height", df


@dataclass
class TitleRasterData:
    fill_col: str
    fill_data_scale: Optional[Scale]
    width: Optional[str]
    height: Optional[str]
    color_scale: Optional["ColorScale"]

    @staticmethod
    def get_height_and_width(
        geom: Geom, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[Optional[str], Optional[str], pd.DataFrame]:

        height, df = _get_height(geom, fs, df)
        width, df = _get_width(geom, fs, df)

        return width, height, df
