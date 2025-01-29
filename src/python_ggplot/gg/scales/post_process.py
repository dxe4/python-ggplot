"""
TODO this file will include many errors,
it will need lot of fixing once everything is in place
Smoothing in particular we may skip for alpha version
"""

from typing import Any, Dict, List, Optional, Tuple, no_type_check

import pandas as pd

from python_ggplot.core.objects import GGException, Scale
from python_ggplot.gg.geom import FilledGeom, Geom, GeomType
from python_ggplot.gg.scales.base import (
    FilledScales,
    GGScale,
    GGScaleContinuous,
    GGScaleDiscrete,
    ScaleType,
)
from python_ggplot.gg.styles import change_style
from python_ggplot.gg.types import GgPlot, GGStyle


def get_scales(
    geom: Geom, filled_scales: FilledScales, y_is_none: bool = False
) -> Tuple[GGScale, Optional[GGScale], List[GGScale]]:
    gid = geom.gg_data.gid

    @no_type_check
    def get_scale(field: Dict[Any, Any]) -> Optional[GGScale]:
        more_scale = [s for s in field["more"] if gid in s.ids]
        if len(more_scale) > 1:
            raise GGException("found more than 1 scale matching gid")
        if len(more_scale) == 1:
            return more_scale[0]
        elif field["main"] is not None:
            return field["main"]
        else:
            return None

    def add_if_any(result_list: List[GGScale], scale: Optional[GGScale]):
        if scale is not None:
            result_list.append(scale)

    x_opt = get_scale(filled_scales.x)
    y_opt = get_scale(filled_scales.y)
    assert x_opt is not None

    other_scales: List[GGScale] = []

    if not y_is_none:
        if y_opt is None:
            raise GGException(
                "The desired geom requires a `y` scale, but none was given. "
                "(Note: The name may differ to the one used in your code as multiple "
                "`geom_*` procedures are mapped to the same kind)"
            )
    elif y_is_none and y_opt is not None:
        raise GGException(
            "The desired geom was given a `y` scale, but none was expected. "
            "(Note: The name may differ to the one used in your code as multiple "
            "`geom_*` procedures are mapped to the same kind)"
        )

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
        add_if_any(other_scales, get_scale(attr_))

    other_scales.extend(filled_scales.facets)

    return x_opt, y_opt, other_scales


def apply_transformations(df: pd.DataFrame, scales: List[GGScale]):
    """
    TODO this will need fixing
    """
    transformations: Dict[Any, Any] = {}

    for scale in scales:
        if scale.scale_type == ScaleType.TRANSFORMED_DATA:
            # TODO formula node logic should be wrong
            col = scale.col.evaluate(df)  # type: ignore
            # This is probably wrong too
            col_str = scale.get_col_name()

            transformations[col_str] = lambda x, s=scale, c=col: s.trans(df[c])  # type: ignore
        else:
            col = scale.col  # type: ignore
            if isinstance(col, str):
                transformations[col] = lambda x, c=col: df[c]  # type: ignore
            else:
                # Assume col is some kind of formula/expression that can be evaluated
                transformations[scale.get_col_name()] = lambda x, c=col: c.evaluate()  # type: ignore

    for col_name, transform_fn in transformations.items():
        df[col_name] = transform_fn(df)


def separate_scales_apply_trafos(
    df: pd.DataFrame, geom: Geom, filled_scales: FilledScales, y_is_none: bool = False
) -> Tuple[GGScale, Optional[GGScale], List[GGScale], List[GGScale]]:
    """
    TODO test this
    """

    x, y, scales = get_scales(geom, filled_scales, y_is_none=y_is_none)

    discretes = [s for s in scales if s.is_discrete()]
    cont = [s for s in scales if not s.is_discrete()]

    discr_cols = list(
        set(s.get_col_name() for s in discretes if s.get_col_name() in df.columns)
    )

    if len(discr_cols) > 0:
        df = df.groupby(discr_cols, group_keys=True)  # type: ignore

    if not y_is_none:
        apply_transformations(df, [x, y] + scales)  # type: ignore
    else:
        apply_transformations(df, [x] + scales)

    return (x, y, discretes, cont)


def split_discrete_set_map(
    df: pd.DataFrame, scales: List[GGScale]
) -> Tuple[List[str], List[str]]:
    set_disc_cols: List[str] = []
    map_disc_cols: List[str] = []

    for scale in scales:
        # TODO this needs 2 fixes, is_column fix and
        # there was an additional check is_consntat
        if scale.gg_data.col.is_column():
            map_disc_cols.append(str(scale.gg_data.col))
        else:
            set_disc_cols.append(str(scale.gg_data.col))

    return set_disc_cols, map_disc_cols


def set_x_attributes(fg: FilledGeom, df: pd.DataFrame, scale: GGScale) -> None:

    if isinstance(scale.gg_data.discrete_kind, GGScaleDiscrete):
        fg.gg_data.num_x = max(fg.gg_data.num_x, df[scale.gg_data.col].nunique())
        fg.gg_data.x_scale = Scale(low=0.0, high=1.0)
        # and assign the label sequence
        # TODO this assumes fg.gg_data.x_discrete_kind = Discrete
        fg.gg_data.x_discrete_kind.label_seq = scale.gg_data.discrete_kind.label_seq  # type: ignore
    elif isinstance(scale.gg_data.discrete_kind, GGScaleContinuous):
        if fg.geom_type != GeomType.RASTER:
            fg.gg_data.num_x = max(fg.gg_data.num_x, len(df))
    else:
        raise GGException("unexpected discrete type")


def apply_cont_scale_if_any(
    yield_df: pd.DataFrame,
    scales: List[GGScale],
    base_style: GGStyle,
    geom_type: GeomType,
    to_clone: bool = False,
):
    result_style = base_style
    result_styles = []
    result_df = yield_df.copy() if to_clone else yield_df

    for scale in scales:
        # TODO col eval is a global issue, fine for now
        result_df[scale.get_col_name()] = scale.col.evaluate(result_df)  # type: ignore

        if scale.scale_type in {ScaleType.TRANSFORMED_DATA, ScaleType.LINEAR_DATA}:
            pass
        else:
            # avoid expensive computation for raster
            if geom_type != GeomType.RASTER:
                # TODO high priority map_data logic is funny overall, add ignore type for now
                sc_vals = scale.map_data(result_df)  # type: ignore
                result_styles = [change_style(base_style, val) for val in sc_vals]  # type: ignore

    if not result_styles:
        result_styles = [base_style]

    return (result_style, result_styles, result_df)


def post_process_scales(filled_scales: FilledGeom, plot: GgPlot):
    x_scale = None
    y_scale = None

    for g in plot.geoms:
        df = (
            g.data if g.data is not None else plot.data.copy(deep=False)
        )  # shallow copy
        filled_geom = None

        if g.kind in ["point", "line", "error_bar", "tile", "text", "raster"]:
            # can be handled the same
            # need x and y data for sure
            if g.stat_kind == "identity":
                filled_geom = filled_identity_geom(df, g, filled_scales)
            elif g.stat_kind == "count":
                filled_geom = filled_count_geom(df, g, filled_scales)
            elif g.stat_kind == "smooth":
                filled_geom = filled_smooth_geom(df, g, filled_scales)
            else:
                filled_geom = filled_bin_geom(df, g, filled_scales)

        elif g.kind in ["histogram", "freq_poly"]:
            if g.stat_kind == "identity":
                # essentially take same data as for point
                filled_geom = filled_identity_geom(df, g, filled_scales)
                # still a histogram like geom, make sure bottom is still at 0!
                filled_geom.y_scale = {
                    "low": min(0.0, filled_geom.y_scale["low"]),
                    "high": filled_geom.y_scale["high"],
                }
            elif g.stat_kind == "bin":
                # calculate histogram
                filled_geom = filled_bin_geom(df, g, filled_scales)
            elif g.stat_kind == "count":
                raise Exception(
                    "For discrete counts of your data use " "`geom_bar` instead!"
                )
            elif g.stat_kind == "smooth":
                raise Exception(
                    "Smoothing statistics not implemented for histogram & frequency polygons. "
                    "Do you want a `density` plot using `geom_density` instead?"
                )

        elif g.kind == "bar":
            if g.stat_kind == "identity":
                # essentially take same data as for point
                filled_geom = filled_identity_geom(df, g, filled_scales)
                # still a geom_bar, make sure bottom is still at 0!
                filled_geom.y_scale = {
                    "low": min(0.0, filled_geom.y_scale["low"]),
                    "high": filled_geom.y_scale["high"],
                }
            elif g.stat_kind == "count":
                # count values in classes
                filled_geom = filled_count_geom(df, g, filled_scales)
            elif g.stat_kind == "bin":
                raise Exception(
                    "For continuous binning of your data use "
                    "`geom_histogram` instead!"
                )
            elif g.stat_kind == "smooth":
                raise Exception(
                    "Smoothing statistics not supported for bar plots. Do you want a "
                    "`density` plot using `geom_density` instead?"
                )

        if x_scale is not None:
            x_scale = merge_scales(x_scale, filled_geom.x_scale)
            y_scale = merge_scales(y_scale, filled_geom.y_scale)
        else:
            x_scale = filled_geom.x_scale
            y_scale = filled_geom.y_scale

        filled_scales.geoms.append(filled_geom)
    final_x_scale, _, _ = calc_tick_locations(x_scale, filled_scales.get_x_ticks())
    final_y_scale, _, _ = calc_tick_locations(y_scale, filled_scales.get_y_ticks())

    filled_scales.x_scale = final_x_scale
    filled_scales.y_scale = final_y_scale
