"""
TODO this file will include many errors,
it will need lot of fixing once everything is in place
Smoothing in particular we may skip for alpha version
"""

from typing import Any, Dict, List, Optional, Tuple, no_type_check

import pandas as pd

from python_ggplot.core.objects import GGException, Scale
from python_ggplot.gg.geom import (
    FilledGeom,
    FilledGeomErrorBar,
    Geom,
    GeomErrorBar,
    GeomType,
)
from python_ggplot.gg.scales.base import (
    FilledScales,
    GGScale,
    GGScaleContinuous,
    GGScaleDiscrete,
    LinearDataScale,
    ScaleType,
    TransformedDataScale,
)
from python_ggplot.gg.styles import change_style, use_or_default
from python_ggplot.gg.types import GgPlot, GGStyle, PositionType
from tests.test_view import AxisKind


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


def add_counts_by_position(
    col_sum: pd.Series[Any], col: pd.Series[Any], pos: PositionType
):
    # TODO use is_numeric_dtype in other places of the code base
    if pd.api.types.is_numeric_dtype(col):  # type: ignore
        if pos == PositionType.STACK:
            if len(col_sum) == 0:
                col_sum = col.copy()
            else:
                col_sum += col
        elif pos in (PositionType.IDENTITY, PositionType.DODGE):
            col_sum = col.copy()
        elif pos == PositionType.FILL:
            col_sum = pd.Series([1.0])
    else:
        col_sum = col.copy()

    return col_sum


def add_zero_keys(
    df: pd.DataFrame, keys: pd.Series[Any], x_col: Any, count_col: str
) -> pd.DataFrame:
    exist_keys = df[x_col].unique()  # type: ignore
    df_zero = pd.DataFrame({x_col: keys[~keys.isin(exist_keys)]})  # type: ignore
    df_zero[count_col] = 0
    return pd.concat([df, df_zero], ignore_index=True)


@no_type_check
def _get_min_max_scale(left, right, operator):
    # TODO CRITICAL reafactor, this is a mess
    # the original code comes from a macro
    # we need to check all macros carefully
    # this wont cause any issues for now
    raise GGException("needs to be ported")


@no_type_check
def _get_height_scale(left, right):
    # TODO CRITICAL reafactor, this is a mess
    # the original code comes from a macro
    # we need to check all macros carefully
    # this wont cause any issues for now
    raise GGException("needs to be ported")


@no_type_check
def _get_width_scale(left, right):
    # TODO CRITICAL reafactor, this is a mess
    # the original code comes from a macro
    # we need to check all macros carefully
    # this wont cause any issues for now
    raise GGException("needs to be ported")


@no_type_check
def get_y_scale(left, right):
    # TODO CRITICAL reafactor, this is a mess
    # the original code comes from a macro
    # we need to check all macros carefully
    # this wont cause any issues for now
    raise GGException("need to be ported")


@no_type_check
def get_x_scale(left, right):
    # TODO CRITICAL reafactor, this is a mess
    # the original code comes from a macro
    # we need to check all macros carefully
    # this wont cause any issues for now
    raise GGException("need to be ported")


@no_type_check
def get_text_scale(left, right):
    # TODO CRITICAL reafactor, this is a mess
    # the original code comes from a macro
    # we need to check all macros carefully
    # this wont cause any issues for now
    raise GGException("need to be ported")


@no_type_check
def get_fill_scale(left, right):
    # TODO CRITICAL reafactor, this is a mess
    # the original code comes from a macro
    # we need to check all macros carefully
    # this wont cause any issues for now
    raise GGException("need to be ported")


def fill_opt_fields(fg: FilledGeom, fs: FilledScales, df: pd.DataFrame):
    """
    TODO CRITICAL
    THIS whole function needs an entire re-write
    we need to provide the functionality very draft
    for now for this to stop being a blocker

    this is the case for most of this file,
    but some things are a lot easier to port

    write unit tests for the original package for this funcs:
        _get_min_max_scale, _get_width_scale and _get_min_max_scale
    """

    def assign_if_any(fg: FilledGeom, scale: Optional[GGScale], attr: Any):
        # TODO this is inherited as tempalte assuming for performanece to avoid func calls
        # we can refactor later
        if scale is not None:
            setattr(fg, attr, scale.get_col_name())

    if fg.geom_type == GeomType.ERROR_BAR:
        # TODO CRITICAL This logic is semi-wrong
        # adding type ignore for now, need to port the macro and write a unit test
        assign_if_any(fg, _get_min_max_scale(fs, fg.gg_data.geom, min), "x_min")  # type: ignore
        assign_if_any(fg, _get_min_max_scale(fs, fg.gg_data.geom, max), "x_max")  # type: ignore
        assign_if_any(fg, _get_min_max_scale(fs, fg.gg_data.geom, min), "y_min")  # type: ignore
        assign_if_any(fg, _get_min_max_scale(fs, fg.gg_data.geom, min), "y_min")  # type: ignore

    elif fg.geom_type in {GeomType.TILE, GeomType.RASTER}:
        h_s = _get_height_scale(fs, fg.gg_data.geom)
        w_s = _get_width_scale(fs, fg.gg_data.geom)
        x_min_s = _get_min_max_scale(fs, fg.gg_data.geom, min)
        x_max_s = _get_min_max_scale(fs, fg.gg_data.geom, max)
        y_min_s = _get_min_max_scale(fs, fg.gg_data.geom, min)
        y_max_s = _get_min_max_scale(fs, fg.gg_data.geom, max)

        if h_s is not None:
            # TODO the type: ignore can go away
            # if we change the if from enum check to isinstance
            # but id rather make something poly morphic on a secondary wave
            # for now just port as is
            fg.data.height = h_s.get_col_name()  # type: ignore
        elif y_min_s is not None and y_max_s is not None:
            min_name = y_min_s.get_col_name()
            max_name = y_max_s.get_col_name()

            # TODO CRITICAL this also comes from macro (non ported yet)
            y_scale = get_y_scale(fs, fg.geom)  # type: ignore
            y_col_name = y_scale.get_col_name()
            df["height"] = df[max_name] - df[min_name]
            df[y_col_name] = df[min_name]
            # TODO the type: ignore can go away, just make it polymorphic instead
            # also not sure why height = some('height') ?????
            fg.gg_data.height = "height"  # type: ignore
        elif y_min_s is not None or y_max_s is not None:
            raise GGException(
                "Invalid combination of aesthetics! If no height given both an `y_min` and `y_max` has to be supplied for geom_{fg.geom_kind}!"
            )
        else:
            if fg.geom_type == GeomType.RASTER:
                col_name = get_y_scale(fs, fg.geom).get_col_name()  # type: ignore
                y_col = df[col_name].unique()  # type: ignore
                # TODO here is the same as before, make polymorphic
                fg.num_y = len(y_col)  # type: ignore
                df["height"] = abs(y_col[1] - y_col[0])
            else:
                print(
                    "INFO: using default height of 1 since no height information supplied. "
                    "Add `height` or (`y_min`, `y_max`) as aesthetics for different values."
                )
                df["height"] = 1.0
            # TODO here is the same as before, make polymorphic
            # also not sure why height = some('height') ?????
            fg.height = "height"  # type: ignore

        # Handle width
        if w_s is not None:
            fg.width = w_s.get_col_name()  # type: ignore
        elif x_min_s is not None and x_max_s is not None:
            min_name = x_min_s.get_col_name()  # type: ignore
            max_name = x_max_s.get_col_name()  # type: ignore
            x_col_name = get_x_scale(fs, fg.geom).get_col_name()  # type: ignore
            df["width"] = df[max_name] - df[min_name]
            df[x_col_name] = df[min_name]
            fg.width = "width"  # type: ignore
        elif x_min_s is not None or x_max_s is not None:
            raise GGException(
                "Invalid combination of aesthetics! If no width given both an `x_min` and `x_max` has to be supplied for geom_{fg.geom_kind}!"
            )
        else:
            if fg.geom_type == GeomType.RASTER:
                x_col = df[get_x_scale(fs, fg.geom).get_col_name()].unique()  # type: ignore
                fg.num_x = len(x_col)  # type: ignore
                df["width"] = abs(x_col[1] - x_col[0])
            else:
                print(
                    "INFO: using default width of 1 since no width information supplied. "
                    "Add `width` or (`x_min`, `x_max`) as aesthetics for different values."
                )
                df["width"] = 1.0
            fg.width = "width"  # type: ignore

        fill_scale = get_fill_scale(fs)  # type: ignore
        if fill_scale is None:
            raise GGException("requires a `fill` aesthetic scale!")
        fg.fill_col = fill_scale.get_col_name()  # type: ignore
        if fill_scale.is_continuous():  # type: ignore
            fg.fill_data_scale = fill_scale.data_scale  # type: ignore
            fg.color_scale = use_or_default(fill_scale.color_scale)  # type: ignore

    elif fg.geom_type == GeomType.TEXT:
        fg.text = str(get_text_scale(fs, fg.geom).col)  # type: ignore

    elif fg.geom_type == GeomType.HISTOGRAM:
        fg.hd_kind = fg.geom.hd_kind  # type: ignore


def encompassing_data_scale(
    scales: List[GGScale],
    axis_kind: AxisKind,
    base_scale: tuple[float, float] = (0.0, 0.0),
) -> Scale:
    result = Scale(low=base_scale[0], high=base_scale[1])

    for scale_ in scales:
        if isinstance(scale_, (LinearDataScale, TransformedDataScale)):
            if scale_.data is not None and scale_.data.axis_kind == axis_kind:
                if isinstance(scale_.gg_data.discrete_kind, GGScaleContinuous):
                    # TODO double check, why does original code not check for continuous?
                    result = result.merge(scale_.gg_data.discrete_kind.data_scale)

    return result


def determine_data_scale(
    scale: GGScale, additional: List[GGScale], df: pd.DataFrame
) -> Scale:

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


def maybe_filter_unique(df: pd.DataFrame, fg: FilledGeom):
    """
    TODO refactor
    """

    if isinstance(fg, FilledGeomErrorBar):
        collect_cols: List[Any] = []
        has_x = False
        has_y = False

        def add_it(field: Any, is_x: Any):
            """
            TODO reafctor this
            """
            nonlocal has_x, has_y
            if field is not None:
                collect_cols.append(field)
                if is_x:
                    has_x = True
                else:
                    has_y = True

        add_it(fg.x_min, True)
        add_it(fg.x_max, True)
        add_it(fg.y_min, False)
        add_it(fg.y_max, False)

        if has_x:
            collect_cols.append(fg.gg_data.y_col)
        if has_y:
            collect_cols.append(fg.gg_data.x_col)

        return df.drop_duplicates(subset=collect_cols)

    return df


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
