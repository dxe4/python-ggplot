from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Type,
    Union,
    cast,
)

import pandas as pd
from typing_extensions import Generator

from python_ggplot.core.objects import AxisKind, GGException, Scale
from python_ggplot.gg.datamancer_pandas_compat import GGValue, VectorCol
from python_ggplot.gg.geom.base import Geom, GeomType, HistogramDrawingStyle
from python_ggplot.gg.geom.tile_raster_data import TitleRasterData
from python_ggplot.gg.types import (
    ColOperator,
    DiscreteKind,
    DiscreteType,
    GGStyle,
    StatType,
)

if TYPE_CHECKING:
    from python_ggplot.gg.scales.base import ColorScale, FilledScales, GGScale


def _optional_scale_col(scale: Optional["GGScale"]) -> Optional[str]:
    if scale is None:
        return None
    return scale.get_col_name()


@dataclass
class FilledGeomData:
    geom: Geom
    x_col: Optional[VectorCol]
    y_col: Optional[VectorCol]
    x_scale: Optional[Scale]
    y_scale: Optional[Scale]
    reversed_x: bool
    reversed_y: bool
    # TODO this logic needs some reworking
    yield_data: OrderedDict[Any, Tuple[GGStyle, List[GGStyle], pd.DataFrame]]
    num_x: int
    num_y: int
    x_discrete_kind: Optional["FilledGeomDiscreteKind"]
    y_discrete_kind: Optional["FilledGeomDiscreteKind"]

    x_transformations: Optional[List[ColOperator]] = None
    y_transformations: Optional[List[ColOperator]] = None

    def is_x_discrete(self):
        return isinstance(self.x_discrete_kind, FilledGeomDiscrete)

    def is_y_discrete(self):
        return isinstance(self.y_discrete_kind, FilledGeomDiscrete)

    def is_x_continuous(self):
        return isinstance(self.x_discrete_kind, FilledGeomContinuous)

    def is_y_continuous(self):
        return isinstance(self.y_discrete_kind, FilledGeomContinuous)


@dataclass
class FilledGeom:
    gg_data: FilledGeomData

    def _ensure_discrete_kind_exists(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            if self.gg_data.x_discrete_kind is None:
                raise GGException("x_discrete_kind is None")
        elif axis_kind == AxisKind.Y:
            if self.gg_data.y_discrete_kind is None:
                raise GGException("y_discrete_kind is None")
        else:
            raise GGException("incorrect axis type")

    def _ensure_x_discrete_kind_exists(self):
        self._ensure_discrete_kind_exists(AxisKind.X)

    def _ensure_y_discrete_kind_exists(self):
        self._ensure_discrete_kind_exists(AxisKind.Y)

    def is_discrete_y(self) -> bool:
        self._ensure_y_discrete_kind_exists()
        return self.gg_data.y_discrete_kind.discrete_type == DiscreteType.DISCRETE  # type: ignore

    def is_discrete_x(self) -> bool:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.discrete_type == DiscreteType.DISCRETE  # type: ignore

    @property
    def discrete_type_y(self) -> DiscreteType:
        self._ensure_y_discrete_kind_exists()
        return self.gg_data.y_discrete_kind.discrete_type  # type: ignore

    @property
    def discrete_type_x(self) -> DiscreteType:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.discrete_type  # type: ignore

    @property
    def discrete_type(self) -> Optional[DiscreteType]:
        self._ensure_x_discrete_kind_exists()
        self._ensure_y_discrete_kind_exists()

        left = self.gg_data.x_discrete_kind.discrete_type  # type: ignore
        right = self.gg_data.y_discrete_kind.discrete_type  # type: ignore
        if left != right:
            return None
        return left

    def get_x_label_seq(self) -> List[GGValue]:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.get_label_seq()  # type: ignore

    def get_y_label_seq(self) -> List[GGValue]:
        self._ensure_y_discrete_kind_exists()
        return self.gg_data.y_discrete_kind.get_label_seq()  # type: ignore

    @property
    def geom_type(self) -> GeomType:
        return self.gg_data.geom.geom_type

    @property
    def stat_type(self) -> StatType:
        return self.gg_data.geom.gg_data.stat_kind.stat_type

    def get_histogram_draw_style(self) -> HistogramDrawingStyle:
        # todo this will be fixed eventually just provide convinience for now
        if not self.geom_type == GeomType.HISTOGRAM:
            raise GGException("attempted to get histogram on non histogram type")
        temp = cast(FilledGeomHistogram, self)
        return temp.histogram_drawing_style

    def enumerate_data(
        self: "FilledGeom",
    ) -> Generator[Tuple[GGValue, GGStyle, List[GGStyle], pd.DataFrame], None, None]:
        for label, tup in self.gg_data.yield_data.items():
            yield label, tup[0], tup[1], tup[2]

    def maybe_filter_unique(self, df: pd.DataFrame) -> pd.DataFrame:
        # this is only needed for FilledGeomErrorBar as of now
        return df


class FilledGeomDiscreteKind(ABC, DiscreteKind):

    @abstractmethod
    def get_label_seq(self) -> List[GGValue]:
        pass

    @property
    @abstractmethod
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.DISCRETE


@dataclass
class FilledGeomDiscrete(FilledGeomDiscreteKind):
    label_seq: List[GGValue]

    def get_label_seq(self) -> List[GGValue]:
        return self.label_seq

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.DISCRETE


@dataclass
class FilledGeomContinuous(FilledGeomDiscreteKind):
    def get_label_seq(self) -> List[GGValue]:
        raise GGException("attempt to get discrete values on continiuous kind")

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.CONTINUOUS


def create_filled_geom_tile_and_raster(
    cls: Union[Type["FilledGeomTitle"], Type["FilledGeomRaster"]],
    geom: Geom,
    fg_data: FilledGeomData,
    fs: "FilledScales",
    df: pd.DataFrame,
) -> Tuple[FilledGeom, pd.DataFrame]:
    from python_ggplot.gg.styles.utils import use_or_default

    fill_data_scale: Optional[Scale] = None
    color_scale: Optional["ColorScale"] = None
    fill_col: str = ""

    width, height, df = TitleRasterData.get_height_and_width(geom, fs, df)

    fill_scale = fs.get_fill_scale(geom)
    if fill_scale is None:
        raise GGException("requires a `fill` aesthetic scale!")

    fill_col = fill_scale.get_col_name()
    if fill_scale.is_continuous():
        fill_data_scale = fill_scale.gg_data.discrete_kind.data_scale  # type: ignore
        # TODO fix this, fine for now
        potential_color_scale: Optional[ColorScale] = getattr(
            fill_scale, "color_scale", None
        )
        color_scale = use_or_default(potential_color_scale)

    tile_raster_data = TitleRasterData(
        fill_col=fill_col,
        fill_data_scale=fill_data_scale,  # type: ignore
        width=width,
        height=height,
        color_scale=color_scale,
    )
    new_fg = cls(gg_data=fg_data, data=tile_raster_data)
    return new_fg, df


@dataclass
class FilledGeomTitle(FilledGeom):
    data: TitleRasterData

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        return create_filled_geom_tile_and_raster(cls, geom, fg_data, fs, df)


@dataclass
class FilledGeomRaster(FilledGeom):
    data: TitleRasterData

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        return create_filled_geom_tile_and_raster(cls, geom, fg_data, fs, df)


@dataclass
class FilledGeomText(FilledGeom):
    text: str

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        new_fg = FilledGeomText(
            gg_data=fg_data,
            text=str(fs.get_text_scale(geom).gg_data.col),  # type: ignore
        )
        return new_fg, df


@dataclass
class FilledGeomHistogram(FilledGeom):
    histogram_drawing_style: HistogramDrawingStyle

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        new_fg = FilledGeomHistogram(
            gg_data=fg_data,
            histogram_drawing_style=geom.histogram_drawing_style,  # type: ignore
        )
        return new_fg, df


@dataclass
class FilledGeomErrorBar(FilledGeom):
    x_min: Optional[str] = None
    y_min: Optional[str] = None
    x_max: Optional[str] = None
    y_max: Optional[str] = None

    def maybe_filter_unique(self, df: pd.DataFrame) -> pd.DataFrame:
        x_values = [i for i in [self.x_min, self.x_max] if i is not None]
        y_values = [i for i in [self.y_min, self.y_max] if i is not None]
        collect_cols = x_values + y_values

        if len(x_values) > 0:
            if self.gg_data.y_col is None:
                raise GGException("expected y_col")
            collect_cols.append(str(self.gg_data.y_col))

        if len(y_values) > 0:
            if self.gg_data.x_col is None:
                raise GGException("expected x_col")
            collect_cols.append(str(self.gg_data.x_col))

        collect_cols = [str(i) for i in collect_cols]
        return df.drop_duplicates(subset=collect_cols)

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        new_fg = FilledGeomErrorBar(
            gg_data=fg_data,
            x_min=_optional_scale_col(fs.get_x_min_scale(geom, optional=True)),
            x_max=_optional_scale_col(fs.get_x_max_scale(geom, optional=True)),
            y_min=_optional_scale_col(fs.get_y_min_scale(geom, optional=True)),
            y_max=_optional_scale_col(fs.get_y_max_scale(geom, optional=True)),
        )
        return new_fg, df


def create_filled_geom(
    fg: FilledGeom, fs: "FilledScales", geom_type: GeomType, df: pd.DataFrame
) -> Tuple[FilledGeom, pd.DataFrame]:
    # Originally "fill_opt_fields"
    # this is a bit ugly, but its lot better than the original
    # its readable enough for now
    if fg.geom_type == GeomType.ERROR_BAR:
        return FilledGeomErrorBar.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    # tile and raster is really the same. fine for now
    elif fg.geom_type == GeomType.TILE:
        return FilledGeomTitle.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    elif fg.geom_type == GeomType.RASTER:
        return FilledGeomRaster.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    elif fg.geom_type == GeomType.TEXT:
        return FilledGeomText.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    elif fg.geom_type == GeomType.HISTOGRAM:
        return FilledGeomHistogram.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    else:
        return fg, df
