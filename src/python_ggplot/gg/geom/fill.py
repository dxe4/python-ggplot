from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Tuple, Type

import pandas as pd
from typing_extensions import Optional

from python_ggplot.core.objects import GGException, Scale
from python_ggplot.gg.constants import (
    SKIP_APPLY_TRANSOFRMATIONS,
    USE_Y_X_MINMAX_AS_X_VALUES,
)
from python_ggplot.gg.geom.base import Geom
from python_ggplot.gg.geom.filled_geom import FilledGeom
from python_ggplot.gg.geom.filled_stat_geom import (
    FilledBinGeom,
    FilledCountGeom,
    FilledIdentityGeom,
    FilledNoneGeom,
    FilledSmoothGeom,
    FilledStatGeom,
)
from python_ggplot.gg.types import GGStyle, StatType
from python_ggplot.graphics.initialize import calc_tick_locations

if TYPE_CHECKING:
    from python_ggplot.gg.scales.base import FilledScales, GGScale, MainAddScales
    from python_ggplot.gg.types import GgPlot


def enumerate_groups(
    df: pd.DataFrame, columns: List[str]
) -> Generator[Tuple[Tuple[Any, ...], pd.DataFrame, Any], None, None]:
    grouped = df.groupby(columns, sort=True)  # type: ignore
    sorted_keys = sorted(grouped.groups.keys(), reverse=True)  # type: ignore
    for keys in sorted_keys:  # type: ignore
        sub_df = grouped.get_group(keys)  # type: ignore
        key_values = list(product(columns, [keys]))  # type: ignore
        yield (keys, sub_df, key_values)


def maybe_inherit_aes(
    sub_df: pd.DataFrame,
    filled_stat_geom: FilledStatGeom,
    filled_geom: FilledGeom,
    style: GGStyle,
    key_values: Any,
) -> GGStyle:
    from python_ggplot.gg.styles.utils import apply_style

    style = deepcopy(style)

    if filled_geom.gg_data.geom.inherit_aes():
        return apply_style(
            style, sub_df, filled_stat_geom.discrete_scales, key_values
        )  # type: ignore
    else:
        return style


def get_scale(geom: Geom, field: Optional["MainAddScales"]) -> Optional["GGScale"]:
    gid = geom.gg_data.gid
    if field is None:
        # TODO is this exception correct?
        raise GGException("attempted to get on empty scale")
    more_scale = [s for s in field.more or [] if gid in s.gg_data.ids]
    if len(more_scale) > 1:
        raise GGException("found more than 1 scale matching gid")
    if len(more_scale) == 1:
        return more_scale[0]
    elif field.main is not None and geom.inherit_aes():
        return field.main
    else:
        return None


def get_scales(
    geom: Geom, filled_scales: "FilledScales", y_is_none: bool = False
) -> Tuple[Optional["GGScale"], Optional["GGScale"], List["GGScale"]]:
    x_opt = get_scale(geom, filled_scales.x)
    y_opt = get_scale(geom, filled_scales.y)

    if x_opt is None:
        x_opt = get_scale(geom, filled_scales.xintercept)

    if y_opt is None:
        y_opt = get_scale(geom, filled_scales.yintercept)

    if y_is_none and y_opt is not None and x_opt is None:
        # TODO high priority
        # if only y is given, we flip the plot
        # this really shouldnt happen, the previous behaviour was that both x and y have to be given
        # so this is a step forward
        # some geom logic is hard coded so that x is the default
        # there is a plan to refactor this soon
        y_opt, x_opt = x_opt, y_opt

    other_scales: List["GGScale"] = []

    attrs_ = [
        filled_scales.color,
        filled_scales.fill,
        filled_scales.size,
        filled_scales.shape,
        filled_scales.x_min,
        filled_scales.x_max,
        filled_scales.y_min,
        filled_scales.y_max,
        filled_scales.width,
        filled_scales.height,
        filled_scales.text,
        filled_scales.y_ridges,
        filled_scales.width,
    ]

    for attr_ in attrs_:
        new_scale = get_scale(geom, attr_)
        if new_scale is not None:
            other_scales.append(new_scale)

    other_scales.extend(filled_scales.facets)
    return x_opt, y_opt, other_scales


def apply_transformations(df: pd.DataFrame, scales: List["GGScale"]):
    """
    TODO this will need fixing
    """
    from python_ggplot.gg.scales.base import ScaleType

    transformations: Dict[Any, Any] = {}
    result: pd.DataFrame = pd.DataFrame()

    for scale in scales:
        if scale.scale_type == ScaleType.TRANSFORMED_DATA:
            # TODO formula node logic should be wrong
            col = scale.col.evaluate(df)  # type: ignore
            # This is probably wrong too
            col_str = scale.get_col_name()

            transformations[col_str] = lambda x, s=scale, c=col: s.trans(df[c])  # type: ignore
        else:
            # TODO this can only be VectorCol for now i think
            # but this is FomrulaNode logic, which we may add in the near future
            col = scale.gg_data.col
            if isinstance(col, str):
                transformations[col] = lambda x, c: x[c]  # type: ignore
            elif isinstance(col, VectorCol):  # type: ignore
                transformations[col.col_name] = lambda x, c: x[c]  # type: ignore
            else:
                # Assume col is some kind of formula/expression that can be evaluated
                transformations[scale.get_col_name()] = lambda x, c: c.evaluate()  # type: ignore

    for col_name, transform_fn in transformations.items():
        result[col_name] = transform_fn(df, col_name)
    return result


def separate_scales_apply_transofrmations(
    df: pd.DataFrame,  # type: ignore
    geom: Geom,
    filled_scales: "FilledScales",
    y_is_none: bool = False,
) -> Tuple[Optional["GGScale"], Optional["GGScale"], List["GGScale"], List["GGScale"]]:
    """
    TODO critical Apply transformations
    """
    x, y, scales = get_scales(geom, filled_scales, y_is_none=y_is_none)

    discretes = [s for s in scales if s.is_discrete()]
    cont = [s for s in scales if s.is_continuous()]

    discr_cols = list(
        set(s.get_col_name() for s in discretes if s.get_col_name() in df.columns)
    )

    if len(discr_cols) > 0:
        df = df.groupby(discr_cols, group_keys=True)  # type: ignore

    # TODO urgent double check this
    # We may not need this until FormulaNode is Implemented

    if not SKIP_APPLY_TRANSOFRMATIONS:
        if not y_is_none:
            apply_transformations(df, [x, y] + scales)  # type: ignore
        else:
            apply_transformations(df, [x] + scales)  # type: ignore

    return (x, y, discretes, cont)


def stat_kind_fg_class(stat_type: StatType) -> Type["FilledStatGeom"]:
    lookup = {
        StatType.IDENTITY: FilledIdentityGeom,
        StatType.COUNT: FilledCountGeom,
        StatType.SMOOTH: FilledSmoothGeom,
        StatType.BIN: FilledBinGeom,
        StatType.DENSITY: FilledBinGeom,
        StatType.NONE: FilledNoneGeom,
    }
    if stat_type not in lookup:
        raise GGException(f"unsuppoerted stat type {stat_type}")

    return lookup[stat_type]


def split_discrete_set_map(
    df: pd.DataFrame, scales: List["GGScale"]  # type: ignore
) -> Tuple[List[str], List[str]]:
    set_disc_cols: List[str] = []
    map_disc_cols: List[str] = []

    for scale in scales:
        if str(scale.gg_data.col) in df.columns:
            if str(scale.gg_data.col) not in map_disc_cols:
                map_disc_cols.append(str(scale.gg_data.col))
        else:
            if str(scale.gg_data.col) not in set_disc_cols:
                set_disc_cols.append(str(scale.gg_data.col))

    return set_disc_cols, map_disc_cols


def create_fillsed_scale_stat_geom(
    df: pd.DataFrame, geom: Any, filled_scales: "FilledScales"
) -> "FilledStatGeom":
    x, y, discrete_scales, continuous_scales = separate_scales_apply_transofrmations(
        df, geom, filled_scales, y_is_none=True
    )
    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discrete_scales)
    filled_stat_geom_cls = stat_kind_fg_class(geom.stat_type)

    xmin = get_scale(geom, filled_scales.x_min)
    xmax = get_scale(geom, filled_scales.x_max)
    ymin = get_scale(geom, filled_scales.y_min)
    ymax = get_scale(geom, filled_scales.y_max)
    if (
        x is None and y is None and None not in [ymax, ymin, xmax, xmin]
    ) and USE_Y_X_MINMAX_AS_X_VALUES:
        x = xmin.merge(xmax)
        y = ymin.merge(ymax)

    fsg = filled_stat_geom_cls(
        geom=geom,
        df=df,
        x=x,
        y=y,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        discrete_scales=discrete_scales,
        continuous_scales=continuous_scales,
        set_discrete_columns=set_disc_cols,
        map_discrete_columns=map_disc_cols,
    )
    return fsg


def create_filled_geom_from_geom(
    df: pd.DataFrame, geom: Geom, filled_scales: "FilledScales"
) -> "FilledGeom":
    if geom.stat_type not in geom.allowed_stat_types:
        raise GGException(
            f"{geom} has stat_type {geom.stat_type} but onle allowed {geom.allowed_stat_types}"
        )

    filled_scale_stat_geom = create_fillsed_scale_stat_geom(df, geom, filled_scales)
    filled_geom, df, style = filled_scale_stat_geom.create_filled_geom(filled_scales)
    filled_scale_stat_geom.post_process(filled_geom, df)
    return filled_geom


def post_process_scales(filled_scales: "FilledScales", plot: "GgPlot"):
    # keeping as is for backwards compatibility for now
    create_filled_geoms_for_filled_scales(filled_scales, plot)


def excpand_scale(scale: Scale, is_continuous: bool):
    """
    an implementation of https://ggplot2.tidyverse.org/reference/expansion.html
    this needs to be configurable, but by default there's an expansion,
    so we add the default one
    need to check how the original ggplot does this
    what is implemented is "fine/good enough for now"
    but for some cases it makes the plots a bit ugly
    maybe the ideal scenario is to expand only if there's elements that go out of the plot
    for example test_geom_linerange
    """

    if not is_continuous:
        return scale

    if scale.low == 0.0:
        return scale

    diff = scale.high - scale.low
    space = diff * 0.1
    scale.low = scale.low - space
    scale.high = scale.high + space

    return scale


def create_filled_geoms_for_filled_scales(
    filled_scales: "FilledScales", plot: "GgPlot"
):
    from python_ggplot.gg.ticks import get_x_ticks, get_y_ticks

    x_scale: Optional[Scale] = None
    y_scale: Optional[Scale] = None

    x_continuous = False
    y_continuous = False

    for geom in plot.geoms:
        if geom.gg_data.data is not None:
            df = geom.gg_data.data
        else:
            df = plot.data.copy(deep=False)

        filled_geom = create_filled_geom_from_geom(df, geom, filled_scales)

        x_continuous = x_continuous or filled_geom.gg_data.is_x_continuous()
        y_continuous = y_continuous or filled_geom.gg_data.is_y_continuous()

        fg_x_scale = filled_geom.gg_data.x_scale
        fg_y_scale = filled_geom.gg_data.x_scale
        if fg_x_scale or fg_y_scale:
            if (
                x_scale is not None
                and not x_scale.is_empty()
                and y_scale is not None
                and not y_scale.is_empty()
            ):
                x_scale = x_scale.merge(filled_geom.gg_data.x_scale)
                if filled_geom.gg_data.y_scale is not None:
                    y_scale = y_scale.merge(filled_geom.gg_data.y_scale)
            else:
                x_scale = filled_geom.gg_data.x_scale
                y_scale = filled_geom.gg_data.y_scale

        filled_scales.geoms.append(filled_geom)

    if x_scale is None or y_scale is None:
        raise GGException("x and y scale have not exist by this point")

    if x_scale.high == x_scale.low:
        print(f"WARNING scale low and high are the same: {x_scale}")
    else:
        final_x_scale, _, _ = calc_tick_locations(x_scale, get_x_ticks(filled_scales))
        final_x_scale = excpand_scale(final_x_scale, x_continuous)
        filled_scales.x_scale = final_x_scale

    if y_scale.high == y_scale.low:
        print(f"WARNING scale low and high are the same: {y_scale}")
    else:
        final_y_scale, _, _ = calc_tick_locations(y_scale, get_y_ticks(filled_scales))
        final_y_scale = excpand_scale(final_y_scale, y_continuous)
        filled_scales.y_scale = final_y_scale
