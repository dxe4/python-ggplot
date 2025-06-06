from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from python_ggplot.gg.datamancer_pandas_compat import VNull, VectorCol
from python_ggplot.gg.geom.base import Geom, GeomType
from python_ggplot.core.objects import AxisKind, GGException, Scale
from python_ggplot.gg.geom.filled_geom import FilledGeomContinuous, FilledGeomData, FilledGeom, FilledGeomDiscrete, FilledGeomDiscreteKind, create_filled_geom
from python_ggplot.gg.types import COUNT_COL, SMOOTH_VALS_COL, ColOperator, GGStyle, gg_col_anonymous, gg_col_const


if TYPE_CHECKING:
    from python_ggplot.gg.scales.base import (
        FilledScales,
        GGScale,
    )


def encompassing_data_scale(
    scales: List["GGScale"],
    axis_kind: AxisKind,
    base_scale: tuple[float, float] = (0.0, 0.0),
) -> Scale:
    from python_ggplot.gg.scales.base import (
        GGScaleContinuous,
        LinearDataScale,
        TransformedDataScale,
    )

    result = Scale(low=base_scale[0], high=base_scale[1])

    for scale_ in scales:
        if isinstance(scale_, (LinearDataScale, TransformedDataScale)):
            if scale_.data is not None and scale_.data.axis_kind == axis_kind:
                if isinstance(scale_.gg_data.discrete_kind, GGScaleContinuous):
                    # TODO double check, why does original code not check for continuous?
                    result = result.merge(scale_.gg_data.discrete_kind.data_scale)

    return result


def determine_data_scale(
    scale: Optional["GGScale"], additional: List["GGScale"], df: pd.DataFrame
) -> Optional[Scale]:
    from python_ggplot.gg.scales.base import GGScaleContinuous, GGScaleDiscrete

    if scale is None:
        return None

    if isinstance(scale.gg_data.col.col_name, (gg_col_const, gg_col_anonymous)):
        return scale.gg_data.col.col_name.get_scale()

    if not str(scale.gg_data.col) in df.columns:
        # TODO, port this logic on formula node
        raise GGException("col not in df")

    if isinstance(scale.gg_data.discrete_kind, GGScaleContinuous):
        # happens for input DFs with 1-2 elements
        existing_scale = scale.gg_data.discrete_kind.data_scale
        if existing_scale is not None:
            return existing_scale
        else:
            low, high = (df[str(scale.gg_data.col)].min(), df[str(scale.col)].max())  # type: ignore
            return Scale(low=low, high=high)  # type: ignore
    elif isinstance(scale.gg_data.discrete_kind, GGScaleDiscrete):
        # for discrete case assign default [0, 1] scale
        return Scale(low=0.0, high=1.0)
    else:
        raise GGException("unexpected discrete kind")


def _get_scale_col_name(scale: Optional["GGScale"]) -> Optional[str]:
    if scale is None:
        return None
    return scale.get_col_name()


def _get_scale_col(scale: Optional["GGScale"]) -> Optional[VectorCol]:
    if scale is None:
        return None
    return scale.gg_data.col


def _get_filled_geom_from_scale(scale: Optional["GGScale"]) -> Optional[FilledGeomDiscreteKind]:
    # todo rename
    if scale is None:
        return None
    return scale.gg_data.discrete_kind.to_filled_geom_kind()


@dataclass
class FilledStatGeom(ABC):
    geom: Geom
    df: pd.DataFrame
    x: Optional["GGScale"]
    y: Optional["GGScale"]
    xmin: Optional["GGScale"]
    ymin: Optional["GGScale"]
    xmax: Optional["GGScale"]
    ymax: Optional["GGScale"]
    discrete_scales: List["GGScale"]
    continuous_scales: List["GGScale"]
    set_discrete_columns: List["str"]
    map_discrete_columns: List["str"]

    @abstractmethod
    def fill_created_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        pass

    def _get_col_transformations(
        self, scale: Optional["GGScale"]
    ) -> Optional[List[ColOperator]]:
        if scale is None:
            return None
        return scale.gg_data.col.get_transformations()

    def create_filled_geom(
        self, filled_scales: "FilledScales"
    ) -> Tuple[FilledGeom, pd.DataFrame, "GGStyle"]:
        from python_ggplot.gg.styles.utils import apply_style

        self.validate()
        fg_data = FilledGeomData(
            geom=self.geom,
            x_col=self.get_x_col(),
            y_col=self.get_y_col(),
            x_scale=self.get_x_scale(),
            y_scale=self.get_y_scale(),
            reversed_x=False,
            reversed_y=False,
            yield_data={},  # type: ignore
            x_discrete_kind=self.get_x_discrete_kind(),
            y_discrete_kind=self.get_y_discrete_kind(),
            num_x=0,
            num_y=0,
            x_transformations=self._get_col_transformations(self.x),
            y_transformations=self._get_col_transformations(self.y),
        )
        fg = FilledGeom(gg_data=fg_data)
        fg, df = create_filled_geom(fg, filled_scales, self.geom.geom_type, self.df)

        style = GGStyle()
        apply_style(
            style,
            df,
            self.discrete_scales,
            [(col, VNull()) for col in self.set_discrete_columns],
        )
        self.geom.gg_data.data = df
        self.df = df
        fg = self.fill_created_geom(filled_scales, fg, style)
        return fg, df, style

    @abstractmethod
    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def get_x_col(self) -> Optional[VectorCol]:
        pass

    @abstractmethod
    def get_y_col(self) -> Optional[VectorCol]:
        pass

    @abstractmethod
    def get_x_scale(self) -> Optional["Scale"]:
        pass

    @abstractmethod
    def get_y_scale(self) -> Optional["Scale"]:
        pass

    @abstractmethod
    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        pass

    @abstractmethod
    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        pass


class FilledNoneGeom(FilledStatGeom):
    def fill_created_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        # TODO clean this up
        x_scale = filled_scales.get_x_scale(filled_geom.gg_data.geom, optional=True)
        y_scale = filled_scales.get_y_scale(filled_geom.gg_data.geom, optional=True)
        if x_scale:
            filled_geom.gg_data.x_scale = (
                x_scale.gg_data.discrete_kind.get_low_level_scale()
            )
            filled_geom.gg_data.x_discrete_kind = (
                x_scale.gg_data.discrete_kind.to_filled_geom_kind()
            )
        else:
            filled_geom.gg_data.x_scale = Scale(low=0.0, high=0.0)

        if y_scale:
            filled_geom.gg_data.y_scale = (
                y_scale.gg_data.discrete_kind.get_low_level_scale()
            )
            filled_geom.gg_data.y_discrete_kind = (
                y_scale.gg_data.discrete_kind.to_filled_geom_kind()
            )
        else:
            filled_geom.gg_data.y_scale = Scale(low=0.0, high=0.0)

        copied_style = deepcopy(style)
        filled_geom.gg_data.yield_data["no_lab_data"] = (
            copied_style,
            [copied_style],
            self.df,
        )
        return filled_geom

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    def validate(self):
        pass

    def get_x_col(self) -> Optional[VectorCol]:
        pass

    def get_y_col(self) -> Optional[VectorCol]:
        pass

    def get_x_scale(self) -> Optional["Scale"]:
        pass

    def get_y_scale(self) -> Optional["Scale"]:
        pass

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        pass

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        pass


class FilledSmoothGeom(FilledStatGeom):
    def fill_created_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        from python_ggplot.gg.geom.utils import filled_smooth_geom

        return filled_smooth_geom(self.df, filled_geom, self, filled_scales, style)

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    def validate(self):
        if self.x.is_discrete():
            raise GGException("expected continuous data")

        if self.y is not None and self.y.is_discrete():
            raise GGException("expected continuous data")

        if self.y is None:
            # TODO i think this logic is wrong, double check
            raise GGException("y is none")

    def get_x_col(self) -> Optional[VectorCol]:
        return self.x.gg_data.col

    def get_y_col(self) -> Optional[VectorCol]:
        return VectorCol(SMOOTH_VALS_COL)

    def get_x_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.x, self.continuous_scales, self.df)

    def get_y_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.y, self.continuous_scales, self.df)

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()


class FilledBinGeom(FilledStatGeom):
    def fill_created_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        from python_ggplot.gg.geom.utils import filled_bin_geom

        return filled_bin_geom(self.df, filled_geom, self, filled_scales, style)

    def count_col(self):
        stat_kind = self.geom.gg_data.stat_kind
        # TODO double check if this was the intention, but i think it is
        if getattr(stat_kind, "density", False):
            return "density"
        else:
            return COUNT_COL

    def width_col(self):
        return "binWidths"

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    def validate(self):
        if self.x.is_discrete():
            raise GGException("For discrete data columns use `geom_bar` instead!")

    def get_x_col(self) -> Optional[VectorCol]:
        return self.x.gg_data.col

    def get_y_col(self) -> Optional[VectorCol]:
        return VectorCol(self.count_col())

    def get_x_scale(self) -> Optional["Scale"]:
        return encompassing_data_scale(self.continuous_scales, AxisKind.X)

    def get_y_scale(self) -> Optional["Scale"]:
        return encompassing_data_scale(self.continuous_scales, AxisKind.Y)

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()


class FilledCountGeom(FilledStatGeom):
    def fill_created_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        from python_ggplot.gg.geom.utils import filled_count_geom

        return filled_count_geom(self.df, filled_geom, self, filled_scales, style)

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    def validate(self):
        if self.x.is_continuous():
            raise GGException(
                "For continuous data columns use `geom_histogram` instead!"
            )

    def get_x_col(self) -> Optional[VectorCol]:
        return self.x.gg_data.col

    def get_y_col(self) -> Optional[VectorCol]:
        return VectorCol(COUNT_COL)

    def get_x_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.x, self.continuous_scales, self.df)

    def get_y_scale(self) -> Optional["Scale"]:
        return encompassing_data_scale(self.continuous_scales, AxisKind.Y)

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        if self.x.is_discrete():
            # TODO critical, easy task
            # double check if we need to pass empty label_seq
            # or if we need x.gg_data.discrete_kind.label_seq
            return FilledGeomDiscrete(label_seq=[])
        else:
            return FilledGeomContinuous()

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()


class FilledIdentityGeom(FilledStatGeom):
    def fill_created_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        from python_ggplot.gg.geom.utils import filled_identity_geom

        return filled_identity_geom(self.df, filled_geom, self, filled_scales, style)

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        if self.geom.geom_type not in {
            GeomType.HISTOGRAM,
            GeomType.FREQ_POLY,
            GeomType.BAR,
        }:
            return

        if fg.gg_data.y_scale is None:
            raise GGException(
                f"expected y scale for geom {self.geom.geom_type} and stat_type IDENTIDY"
            )

        fg.gg_data.y_scale = Scale(
            low=min(0.0, fg.gg_data.y_scale.low),
            high=fg.gg_data.y_scale.high,
        )

    def validate(self):
        pass

    def get_x_col(self) -> Optional[VectorCol]:
        return _get_scale_col(self.x)

    def get_y_col(self) -> Optional[VectorCol]:
        return _get_scale_col(self.y)

    def get_x_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.x, self.continuous_scales, self.df)

    def get_y_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.y, self.continuous_scales, self.df)

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return _get_filled_geom_from_scale(self.x)

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return _get_filled_geom_from_scale(self.y)
