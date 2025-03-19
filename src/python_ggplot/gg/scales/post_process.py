"""
TODO this file will include many errors,
it will need lot of fixing once everything is in place
Smoothing in particular we may skip for alpha version
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_ggplot.common.maths import histogram
from python_ggplot.core.objects import AxisKind, GGException, Scale
from python_ggplot.gg.datamancer_pandas_compat import VectorCol, VNull, VString
from python_ggplot.gg.geom.base import (
    FilledGeom,
    FilledGeomContinuous,
    FilledGeomData,
    FilledGeomDiscrete,
    FilledGeomErrorBar,
    Geom,
    GeomType,
    HistogramDrawingStyle,
)
from python_ggplot.gg.scales.base import (
    FilledScales,
    GGScale,
    GGScaleContinuous,
    GGScaleDiscrete,
    LinearDataScale,
    MainAddScales,
    ScaleType,
    TransformedDataScale,
)
from python_ggplot.gg.styles.utils import apply_style, change_style, use_or_default
from python_ggplot.gg.ticks import get_x_ticks, get_y_ticks
from python_ggplot.gg.types import (
    COUNT_COL,
    PREV_VALS_COL,
    SMOOTH_VALS_COL,
    BinByType,
    GgPlot,
    GGStyle,
    PositionType,
    SmoothMethodType,
    StatBin,
    StatSmooth,
    StatType,
)
from python_ggplot.graphics.initialize import calc_tick_locations


def get_scales(
    geom: Geom, filled_scales: FilledScales, y_is_none: bool = False
) -> Tuple[GGScale, Optional[GGScale], List[GGScale]]:
    gid = geom.gg_data.gid

    def get_scale(field: Optional[MainAddScales]) -> Optional[GGScale]:
        if field is None:
            # TODO is this exception correct?
            raise GGException("attempted to get on empty scale")
        more_scale = [s for s in field.more or [] if gid in s.gg_data.ids]
        if len(more_scale) > 1:
            raise GGException("found more than 1 scale matching gid")
        if len(more_scale) == 1:
            return more_scale[0]
        elif field.main is not None:
            return field.main
        else:
            return None

    x_opt = get_scale(filled_scales.x)
    y_opt = get_scale(filled_scales.y)
    if x_opt is None:
        raise GGException("x_opt is None")

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
        new_scale = get_scale(attr_)
        if new_scale is not None:
            other_scales.append(new_scale)

    other_scales.extend(filled_scales.facets)
    return x_opt, y_opt, other_scales


def apply_transformations(df: pd.DataFrame, scales: List[GGScale]):
    """
    TODO this will need fixing
    """
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
    filled_scales: FilledScales,
    y_is_none: bool = False,
) -> Tuple[GGScale, Optional[GGScale], List[GGScale], List[GGScale]]:
    """
    TODO test this
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

    # if not y_is_none:
    #     apply_transformations(df, [x, y] + scales)  # type: ignore
    # else:
    #     apply_transformations(df, [x] + scales)

    return (x, y, discretes, cont)


def split_discrete_set_map(
    df: pd.DataFrame, scales: List[GGScale]  # type: ignore
) -> Tuple[List[str], List[str]]:
    set_disc_cols: List[str] = []
    map_disc_cols: List[str] = []

    for scale in scales:
        # TODO URGENT easy fix
        # Original implementation checks if its constant
        if str(scale.gg_data.col) in df.columns:
            map_disc_cols.append(str(scale.gg_data.col))
        else:
            set_disc_cols.append(str(scale.gg_data.col))

    return set_disc_cols, map_disc_cols


def set_x_attributes(fg: FilledGeom, df: pd.DataFrame, scale: GGScale) -> None:

    if isinstance(scale.gg_data.discrete_kind, GGScaleDiscrete):
        fg.gg_data.num_x = max(fg.gg_data.num_x, df[str(scale.gg_data.col)].nunique())
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
    col_sum: pd.Series,  # type: ignore
    col: pd.Series,  # type: ignore
    pos: Optional[PositionType],
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
    df: pd.DataFrame, keys: pd.Series, x_col: Any, count_col: str
) -> pd.DataFrame:
    exist_keys = df[x_col].unique()  # type: ignore
    df_zero = pd.DataFrame({x_col: keys[~keys.isin(exist_keys)]})  # type: ignore
    df_zero[count_col] = 0
    return pd.concat([df, df_zero], ignore_index=True)


# TODO the following functions are repetitive
# we can make something more re-usable
# we keep them for now for backwards compat
# the original ones created by macro


def _get_y_max_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.y_max, geom=geom, optional=optional
    )


def _get_y_min_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.y_min, geom=geom, optional=optional
    )


def _get_x_max_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.x_max, geom=geom, optional=optional
    )


def _get_x_min_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.x_min, geom=geom, optional=optional
    )


def _get_height_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.height, geom=geom, optional=optional
    )


def _get_width_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.width, geom=geom, optional=optional
    )


def get_y_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(attr=filled_scales.y, geom=geom, optional=optional)


def get_x_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(attr=filled_scales.x, geom=geom, optional=optional)


def get_text_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.text, geom=geom, optional=optional
    )


def get_fill_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.fill, geom=geom, optional=optional
    )


def get_weight_scale(
    filled_scales: FilledScales, geom: Geom, optional: bool = False
) -> Optional[GGScale]:
    return filled_scales.get_scale(
        attr=filled_scales.weight, geom=geom, optional=optional
    )


def fill_opt_fields(fg: FilledGeom, fs: FilledScales, df: pd.DataFrame):
    def assign_if_any(fg: FilledGeom, scale: Optional[GGScale], attr: Any):
        # TODO this is inherited as tempalte assuming for performanece to avoid func calls
        # we can refactor later
        if scale is not None:
            setattr(fg, attr, scale.get_col_name())

    if fg.geom_type == GeomType.ERROR_BAR:
        assign_if_any(fg, _get_x_min_scale(fs, fg.gg_data.geom), "x_min")
        assign_if_any(fg, _get_x_max_scale(fs, fg.gg_data.geom), "x_max")
        assign_if_any(fg, _get_y_min_scale(fs, fg.gg_data.geom), "y_min")
        assign_if_any(fg, _get_y_max_scale(fs, fg.gg_data.geom), "y_max")

    elif fg.geom_type in {GeomType.TILE, GeomType.RASTER}:
        h_s = _get_height_scale(fs, fg.gg_data.geom)
        w_s = _get_width_scale(fs, fg.gg_data.geom)
        x_min_s = _get_x_min_scale(fs, fg.gg_data.geom)
        x_max_s = _get_x_max_scale(fs, fg.gg_data.geom)
        y_min_s = _get_y_min_scale(fs, fg.gg_data.geom)
        y_max_s = _get_y_max_scale(fs, fg.gg_data.geom)

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


def filled_identity_geom(
    df: pd.DataFrame, geom: Geom, filled_scales: FilledScales
) -> FilledGeom:
    """
    TODO refactor/test/fix this
    """
    x, y, discretes, cont = separate_scales_apply_transofrmations(
        df, geom, filled_scales
    )
    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discretes)
    if y is None:
        # TODO double check if this is correct
        raise GGException("y is None")

    fg_data = FilledGeomData(
        geom=geom,
        x_col=x.get_col_name(),
        y_col=y.get_col_name(),
        x_scale=determine_data_scale(x, cont, df),
        y_scale=determine_data_scale(y, cont, df),
        reversed_x=False,
        reversed_y=False,
        yield_data={},  # type: ignore
        x_discrete_kind=x.gg_data.discrete_kind.to_filled_geom_kind(),
        y_discrete_kind=y.gg_data.discrete_kind.to_filled_geom_kind(),
        num_x=0,
        num_y=0,
    )

    result = FilledGeom(gg_data=fg_data)
    # TODO refactor
    fill_opt_fields(result, filled_scales, df)

    # TODO this has to change, but is fine for now
    style = GGStyle()

    # Apply style for set values
    for set_val in set_disc_cols:
        # TODO
        # this may be bug in the original code that loops through set_disc_cols twice
        # (hence the unused varialbe)
        # highly probably this is the case
        # for setVal in setDiscCols:
        #    applyStyle(style, df, discretes, setDiscCols.mapIt((it, Value(kind: VNull))))
        # we should double check and make a PR to the nim package if thats the case
        # seems extra computation but probably no visual issues
        apply_style(style, df, discretes, [(col, VNull()) for col in set_disc_cols])

    if len(map_disc_cols) > 0:
        grouped = df.groupby(map_disc_cols, sort=True)  # type: ignore
        col = pd.Series(dtype=float)  # type: ignore

        # TODO this needs fixing, ignore types for now, keep roughly working logic
        for keys, sub_df in grouped:  # type: ignore
            if len(keys) > 1:
                raise GGException("we assume this is 1")

            apply_style(style, sub_df, discretes, [(keys[0], VString(i)) for i in grouped.groups])  # type: ignore

            yield_df = sub_df.copy()
            set_x_attributes(result, yield_df, x)

            if geom.gg_data.position == PositionType.STACK:
                yield_df[PREV_VALS_COL] = 0.0 if len(col) == 0 else col.copy()  # type: ignore

            col = add_counts_by_position(
                yield_df[result.gg_data.y_col],  # type: ignore
                col,  # type: ignore
                geom.gg_data.position,
            )

            if geom.gg_data.position == PositionType.STACK and not (
                (
                    geom.geom_type == GeomType.HISTOGRAM
                    and geom.gg_data.histogram_drawing_style
                    == HistogramDrawingStyle.BARS
                )
                or (geom.geom_type == GeomType.BAR)
            ):
                yield_df[result.gg_data.y_col] = col

            yield_df = maybe_filter_unique(yield_df, result)
            style_, styles_, temp_yield_df = apply_cont_scale_if_any(
                yield_df, cont, style, geom.geom_type, to_clone=True
            )
            result.gg_data.yield_data[keys] = (style_, styles_, temp_yield_df)  # type: ignore

        if geom.gg_data.position == PositionType.STACK and result.is_discrete_y():
            result.gg_data.y_scale = result.gg_data.y_scale.merge(
                Scale(low=result.gg_data.y_scale.low, high=col.max())  # type: ignore
            )

        if (
            geom.geom_type == GeomType.HISTOGRAM
            and geom.gg_data.position == PositionType.STACK
            and geom.gg_data.histogram_drawing_style == HistogramDrawingStyle.OUTLINE
        ):
            result.gg_data.yield_data = dict(reversed(list(result.gg_data.yield_data.items())))  # type: ignore
    else:
        yield_df = df.copy()
        yield_df[PREV_VALS_COL] = 0.0
        yield_df = maybe_filter_unique(yield_df, result)
        set_x_attributes(result, yield_df, x)
        key = ("", None)
        result.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, style, geom.geom_type
        )

    if y.is_discrete():
        # TODO fix
        # y.label_seqwill exist since is discrete, but this needs refactor anyway
        result.gg_data.y_discrete_kind.label_seq = y.gg_data.discrete_kind.label_seq  # type: ignore

    result.gg_data.num_y = result.gg_data.num_x
    return result


def call_smoother(
    fg: FilledGeom, df: pd.DataFrame, scale: GGScale, range: Any
) -> NDArray[Any]:

    geom = fg.gg_data.geom
    stat_kind = geom.gg_data.stat_kind
    if not isinstance(stat_kind, StatSmooth):
        raise GGException("stat type has to be smooth to call smooth function")

    data = df[scale.get_col_name()]  # type: ignore

    if stat_kind.method_type == SmoothMethodType.SVG:
        # TODO we need to convert the result to np.array float
        # smoothing is lower priority, so for now we are fine without it
        return stat_kind.svg_smooth(data)  # type: ignore

    elif stat_kind.method_type == SmoothMethodType.POLY:
        x_data = df[fg.gg_data.x_col]  # type: ignore
        return stat_kind.polynomial_smooth(x_data, data)  # type: ignore

    elif stat_kind.method_type == SmoothMethodType.LM:
        raise GGException("Levenberg-Marquardt fitting is not implemented yet.")

    raise GGException("Unknown smoothing method")


def filled_smooth_geom(
    df: pd.DataFrame, geom: Geom, filled_scales: FilledScales
) -> FilledGeom:
    """
    TODO complete refactor
    reuse logic with filled_identity_geom
    doesnt make a difference for now,
    need a draft version of this to get all the unit tests running
    """

    x, y, discretes, cont = separate_scales_apply_transofrmations(
        df, geom, filled_scales
    )
    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discretes)

    if x.is_discrete():
        raise GGException("expected continuous data")
    if y is not None and y.is_discrete():
        raise GGException("expected continuous data")

    if y is None:
        # TODO i think this logic is wrong, double check
        raise GGException("y is none")

    fg_data = FilledGeomData(
        geom=geom,
        x_col=x.get_col_name(),
        y_col=SMOOTH_VALS_COL,
        x_scale=determine_data_scale(x, cont, df),
        y_scale=determine_data_scale(y, cont, df),
        x_discrete_kind=FilledGeomContinuous(),
        y_discrete_kind=FilledGeomContinuous(),
        # not explicitly passed at initialisisation, we set some defaults
        # TODO investiage if needed
        reversed_x=False,
        reversed_y=False,
        yield_data={},  # type: ignore
        num_x=0,
        num_y=0,
    )
    result = FilledGeom(gg_data=fg_data)
    fill_opt_fields(result, filled_scales, df)

    style = GGStyle()
    for set_val in set_disc_cols:
        # same with filled_identity_geom
        # this may be a bug
        apply_style(style, df, discretes, [(col, VNull()) for col in set_disc_cols])

    if len(map_disc_cols) > 0:
        df = df.groupby(map_disc_cols)  # type: ignore
        col = pd.Series(dtype=float)  # type: ignore

        # TODO CRITICAL deal with pandas multi index...
        # assume this works for now to finish the rest
        for keys, sub_df in df.sort_values(ascending=False):  # type: ignore
            apply_style(style, sub_df, discretes, keys)  # type: ignore
            yield_df = sub_df.copy()  # type: ignore

            smoothed = call_smoother(
                result,
                yield_df,  # type: ignore
                y,
                range=x.gg_data.discrete_kind.data_scale,  # type: ignore
            )
            yield_df[SMOOTH_VALS_COL] = smoothed
            set_x_attributes(result, yield_df, x)  # type: ignore

            if geom.gg_data.position == PositionType.STACK:
                yield_df[PREV_VALS_COL] = pd.Series(0.0, index=yield_df.index) if len(col) == 0 else col.copy()  # type: ignore

            # possibly modify `col` if stacking
            add_counts_by_position(
                yield_df[result.y_col],  # type: ignore
                col,  # type: ignore
                geom.gg_data.position,
            )

            if geom.gg_data.position == PositionType.STACK and not (
                (
                    geom.geom_type == GeomType.HISTOGRAM
                    and geom.gg_data.histogram_drawing_style
                    == HistogramDrawingStyle.BARS
                )
                or (geom.geom_type == GeomType.BAR)
            ):
                yield_df[result.y_col] = col  # type: ignore

            yield_df = maybe_filter_unique(yield_df, result)  # type: ignore
            result.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
                yield_df, cont, style, geom.geom_type, to_clone=True  # type: ignore
            )

        if geom.gg_data.position == PositionType.STACK and not result.is_discrete_y():
            # only update required if stacking, as we've computed the range beforehand
            result.gg_data.y_scale = result.gg_data.y_scale.merge(
                Scale(low=result.gg_data.y_scale.low, high=col.max())  # type: ignore
            )

        if (
            geom.geom_type == GeomType.HISTOGRAM
            and geom.gg_data.position == PositionType.STACK
            and geom.gg_data.histogram_drawing_style == HistogramDrawingStyle.OUTLINE
        ):
            result.gg_data.yield_data = dict(reversed(list(result.gg_data.yield_data.items())))  # type: ignore
    else:
        yield_df = df.copy()
        smoothed = call_smoother(
            result, yield_df, y, range=x.data_scale  # type: ignore TODO critical FIX
        )
        yield_df[PREV_VALS_COL] = pd.Series(0.0, index=yield_df.index)  # type: ignore
        yield_df[SMOOTH_VALS_COL] = smoothed
        yield_df = maybe_filter_unique(yield_df, result)
        set_x_attributes(result, yield_df, x)
        key = ("", VNull())
        result.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, style, geom.geom_type
        )

    result.gg_data.num_y = result.gg_data.num_x

    return result


def call_histogram(
    geom: Geom,
    df: pd.DataFrame,
    scale: GGScale,
    weight_scale: Optional[GGScale],
    range_scale: Scale,
) -> Tuple[
    List[float],
    List[float],
    List[float],
]:
    """
    TODO revisti this once public interface is ready
    """
    stat_kind = geom.gg_data.stat_kind
    if not isinstance(stat_kind, StatBin):
        raise GGException("expected bin stat type")

    def read_tmpl(sc: Any):
        return sc.col.evaluate(df).to_numpy(dtype=float)

    data = read_tmpl(scale)
    hist = []
    bin_edges = []
    bin_widths = []

    def call_hist(bins_arg: Any):
        """
        TODO refactor this
        """
        if stat_kind.bin_by == BinByType.FULL:
            range_val = (range_scale.low, range_scale.high)
        else:
            range_val = (0.0, 0.0)

        weight_data = read_tmpl(weight_scale) or []  # type: ignore

        nonlocal hist, bin_edges
        hist, bin_edges = histogram(
            data,
            bins_arg,
            weights=weight_data or None,  # type: ignore
            range=range_val,
            density=stat_kind.density,
        )

    if stat_kind.bin_edges is not None:
        call_hist(stat_kind.bin_edges)
    elif stat_kind.bin_width is not None:
        bins = round((range_scale.high - range_scale.low) / stat_kind.bin_width)
        call_hist(int(bins))
    else:
        call_hist(stat_kind.num_bins)

    bin_widths = np.diff(bin_edges)  # type: ignore
    hist = np.append(hist, 0.0)  # type: ignore
    return hist, bin_edges, bin_widths  # type: ignore


def filled_bin_geom(df: pd.DataFrame, geom: Geom, filled_scales: FilledScales):
    """
    todo refactor the whole function and re use the code
    """

    stat_kind = geom.gg_data.stat_kind
    # TODO double check if this was the intention, but i think it is
    if getattr(stat_kind, "density", False):
        count_col = "density"
    else:
        count_col = "COUNT"

    width_col = "binWidths"

    x, _, discretes, cont = separate_scales_apply_transofrmations(
        df, geom, filled_scales, y_is_none=True
    )

    if x.is_discrete():
        raise GGException("For discrete data columns use `geom_bar` instead!")

    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discretes)

    fg_data = FilledGeomData(
        geom=geom,  # we could do a deep copy on this
        x_col=x.get_col_name(),
        y_col=count_col,
        x_scale=encompassing_data_scale(cont, AxisKind.X),
        y_scale=encompassing_data_scale(cont, AxisKind.Y),
        # not explicitly passed at initialisisation, we set some defaults
        # TODO investiage if needed
        x_discrete_kind=FilledGeomContinuous(),
        y_discrete_kind=FilledGeomContinuous(),
        reversed_x=False,
        reversed_y=False,
        yield_data={},  # type: ignore
        num_x=0,
        num_y=0,
    )
    result = FilledGeom(gg_data=fg_data)

    fill_opt_fields(result, filled_scales, df)

    style = GGStyle()
    for set_val in set_disc_cols:
        apply_style(style, df, discretes, [(col, VNull()) for col in set_disc_cols])

    if map_disc_cols:
        df = df.group_by(map_disc_cols)  # type: ignore TODO
        col = pd.Series(dtype=float)

        for keys, sub_df in df.sort_values(ascending=False):  # type: ignore
            # now consider settings
            apply_style(style, sub_df, discretes, keys)  # type: ignore
            # before we assign calculate histogram
            hist, bins, _ = call_histogram(
                geom,
                sub_df,  # type: ignore
                x,
                get_weight_scale(filled_scales, geom),
                x.gg_data.discrete_kind.data_scale,  # type: ignore TODO
            )

            yield_df = pd.DataFrame({x.get_col_name(): bins, count_col: hist})

            if geom.gg_data.position == PositionType.STACK:
                yield_df["PREV_VALS"] = col if len(col) > 0 else 0.0

            add_counts_by_position(col, pd.Series(hist), geom.gg_data.position)

            if geom.gg_data.position == PositionType.STACK:
                if not (
                    (
                        geom.geom_type == GeomType.HISTOGRAM
                        and geom.gg_data.histogram_drawing_style
                        == HistogramDrawingStyle.BARS
                    )
                    or (geom.geom_type == GeomType.BAR)
                ):
                    yield_df[result.gg_data.y_col] = col

            yield_df = maybe_filter_unique(yield_df, result)  # type: ignore
            result.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
                yield_df, cont, style, geom.geom_type, to_clone=True  # type: ignore
            )

            result.gg_data.num_x = max(result.gg_data.num_x, len(yield_df))

            if geom.geom_type == GeomType.FREQ_POLY:
                bin_width = float(bins[1] - bins[0]) if len(bins) > 1 else 0.0
                result.gg_data.x_scale = result.gg_data.x_scale.merge(
                    Scale(
                        low=float(min(bins)) - bin_width / 2.0,
                        high=float(max(bins)) + bin_width / 2.0,
                    )
                )
            else:
                result.gg_data.x_scale = result.gg_data.x_scale.merge(
                    Scale(low=float(min(bins)), high=float(max(bins)))
                )

            result.gg_data.y_scale = result.gg_data.y_scale.merge(
                Scale(low=0.0, high=float(col.max()))  # type: ignore
            )
    else:
        hist, bins, bin_widths = call_histogram(
            geom,
            df,
            x,
            get_weight_scale(filled_scales, geom),
            x.gg_data.discrete_kind.data_scale,  # type: ignore TODO
        )

        yield_df = pd.DataFrame(
            {x.get_col_name(): bins, count_col: hist, width_col: bin_widths}
        )
        yield_df["PREV_VALS"] = 0.0
        yield_df = maybe_filter_unique(yield_df, result)  # type: ignore

        key = ("", VNull())

        if len(cont) != 0:
            raise GGException("seems the data is discrete")

        result.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, style, geom.geom_type
        )
        result.gg_data.num_x = len(yield_df)
        result.gg_data.x_scale = result.gg_data.x_scale.merge(
            Scale(low=float(min(bins)), high=float(max(bins)))
        )
        result.gg_data.y_scale = result.gg_data.y_scale.merge(
            Scale(low=0.0, high=float(max(hist)))
        )

    result.gg_data.num_y = round(result.gg_data.y_scale.high)

    if x.is_discrete():
        # TODO fix, this is an error
        result.gg_data.x_label_seq = x.gg_data.label_seq  # type: ignore

    return result


def count_(
    df: pd.DataFrame,  # type: ignore
    x_col: str,
    name: str,
    weights: Optional[GGScale] = None,
) -> pd.DataFrame:
    # TODO critical, medium complexity
    # we rename to counts_GGPLOTNIM_INTERNAL
    # need to make a choice here
    if weights is None:
        result = df[x_col].value_counts().reset_index()
        result = result[[x_col, "count"]].rename({"count": name})
    else:
        result = df.groupby(x_col)[weights.get_col_name()].sum().reset_index()  # type: ignore
        result = result[[x_col, "count"]].rename({"count": name})

    return result


def filled_count_geom(df: pd.DataFrame, geom: Any, filled_scales: Any) -> FilledGeom:
    """
    todo refactor the whole function and re use the code
    """
    x, _, discretes, cont = separate_scales_apply_transofrmations(
        df, geom, filled_scales, y_is_none=True
    )

    if x.is_continuous():
        raise ValueError("For continuous data columns use `geom_histogram` instead!")

    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discretes)
    x_col = x.get_col_name()

    if x.is_discrete():
        # TODO critical, easy task
        # double check if we need to pass empty label_seq
        # or if we need x.gg_data.discrete_kind.label_seq
        x_discrete_kind = FilledGeomDiscrete(label_seq=[])
    else:
        x_discrete_kind = FilledGeomContinuous()

    fg_data = FilledGeomData(
        geom=geom,
        x_col=x_col,
        y_col=COUNT_COL,
        x_scale=determine_data_scale(x, cont, df),
        y_scale=encompassing_data_scale(cont, AxisKind.Y),
        # not explicitly passed at initialisisation, we set some defaults
        # TODO investiage if needed
        x_discrete_kind=x_discrete_kind,
        y_discrete_kind=FilledGeomContinuous(),
        reversed_x=False,
        reversed_y=False,
        yield_data={},  # type: ignore
        num_x=0,
        num_y=0,
    )
    result = FilledGeom(gg_data=fg_data)

    fill_opt_fields(result, filled_scales, df)

    all_classes = df[x_col].unique()  # type: ignore
    style = GGStyle()

    # TODO bug in original code? set_val is not used
    # it does another loop inside
    for set_val in set_disc_cols:
        apply_style(style, df, discretes, [(col, VNull()) for col in set_disc_cols])

    if len(map_disc_cols) > 0:
        grouped = df.groupby(map_disc_cols, sort=False)  # type: ignore
        col = pd.Series(dtype=float)  # For stacking

        if len(cont) > 0:
            raise GGException("cont >0")

        for keys, sub_df in grouped:  # type: ignore
            apply_style(style, sub_df, discretes, keys)  # type: ignore

            weight_scale = get_weight_scale(filled_scales, geom)
            yield_df = count_(sub_df, x_col, "", weight_scale)

            add_zero_keys(yield_df, all_classes, x_col, "count")  # type: ignore
            yield_df = yield_df.sort_values(x_col)  # type: ignore

            if geom.gg_data.position == PositionType.STACK:
                yield_df["prev_vals"] = 0.0 if len(col) == 0 else col.copy()

            col = add_counts_by_position(
                col, yield_df["count"], geom.position  # type: ignore
            )

            if geom.gg_data.position == PositionType.STACK and not (
                (
                    geom.geom_type == GeomType.HISTOGRAM
                    and geom.gg_data.histogram_drawing_style
                    == HistogramDrawingStyle.BARS
                )
                or (geom.geom_type == GeomType.BAR)
            ):
                yield_df["count"] = col

            maybe_filter_unique(yield_df, result)

            result.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
                yield_df, cont, style, geom.kind, to_clone=True
            )

            set_x_attributes(result, yield_df, x)
            result.gg_data.y_scale = result.gg_data.y_scale.merge(
                Scale(low=0.0, high=float(col.max()))  # type: ignore
            )
    else:
        if len(cont) > 0:
            raise GGException("cont > 0")

        weight_scale = get_weight_scale(filled_scales, geom, optional=True)
        yield_df = count_(df, x_col, COUNT_COL, weight_scale)
        # TODO double check prev_vals
        yield_df[PREV_VALS_COL] = 0.0

        key = ("", VNull())
        yield_df = maybe_filter_unique(yield_df, result)
        result.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, style, geom.geom_type
        )
        set_x_attributes(result, yield_df, x)
        result.gg_data.y_scale = result.gg_data.y_scale.merge(
            Scale(low=0.0, high=float(yield_df[COUNT_COL].max()))  # type: ignore
        )

    result.gg_data.num_y = round(result.gg_data.y_scale.high)
    result.gg_data.num_x = len(all_classes)  # type: ignore

    if result.gg_data.num_x != len(all_classes):  # type: ignore
        # todo provide better messages...
        raise GGException("ERROR")

    return result


def post_process_scales(filled_scales: FilledScales, plot: GgPlot):
    """
    TODO refactor this#
    we need something like geom.fill() with mixins
    make it work first
    """
    x_scale: Optional[Scale] = None
    y_scale: Optional[Scale] = None

    for geom in plot.geoms:
        geom_data = geom.gg_data.data
        geom_data = geom_data or plot.data.copy(deep=False)
        df = geom_data
        filled_geom = None

        if geom.geom_type in [
            GeomType.POINT,
            GeomType.LINE,
            GeomType.ERROR_BAR,
            GeomType.TILE,
            GeomType.TEXT,
            GeomType.RASTER,
        ]:
            # can be handled the same
            # need x and y data for sure
            if geom.stat_type == StatType.IDENTITY:
                filled_geom = filled_identity_geom(df, geom, filled_scales)
            elif geom.stat_type == StatType.COUNT:
                filled_geom = filled_count_geom(df, geom, filled_scales)
            elif geom.stat_type == StatType.SMOOTH:
                filled_geom = filled_smooth_geom(df, geom, filled_scales)
            else:
                filled_geom = filled_bin_geom(df, geom, filled_scales)

        elif geom.geom_type in [GeomType.HISTOGRAM, GeomType.FREQ_POLY]:
            if geom.stat_type == StatType.IDENTITY:
                # essentially take same data as for point
                filled_geom = filled_identity_geom(df, geom, filled_scales)
                # still a histogram like geom, make sure bottom is still at 0!
                filled_geom.gg_data.y_scale = Scale(
                    low=min(0.0, filled_geom.gg_data.y_scale.low),
                    high=filled_geom.gg_data.y_scale.high,
                )
            elif geom.stat_type == StatType.BIN:
                # calculate histogram
                filled_geom = filled_bin_geom(df, geom, filled_scales)
            elif geom.stat_type == StatType.COUNT:
                raise Exception(
                    "For discrete counts of your data use " "`geom_bar` instead!"
                )
            elif geom.stat_type == StatType.SMOOTH:
                raise Exception(
                    "Smoothing statistics not implemented for histogram & frequency polygons. "
                    "Do you want a `density` plot using `geom_density` instead?"
                )

        elif geom.geom_type == GeomType.BAR:
            if geom.stat_type == StatType.IDENTITY:
                # essentially take same data as for point
                filled_geom = filled_identity_geom(df, geom, filled_scales)
                # still a geom_bar, make sure bottom is still at 0!
                filled_geom.gg_data.y_scale = Scale(
                    low=min(0.0, filled_geom.gg_data.y_scale.low),
                    high=filled_geom.gg_data.y_scale.high,
                )
            elif geom.stat_type == StatType.COUNT:
                # count values in classes
                filled_geom = filled_count_geom(df, geom, filled_scales)
            elif geom.stat_type == StatType.BIN:
                raise Exception(
                    "For continuous binning of your data use "
                    "`geom_histogram` instead!"
                )
            elif geom.stat_type == StatType.SMOOTH:
                raise Exception(
                    "Smoothing statistics not supported for bar plots. Do you want a "
                    "`density` plot using `geom_density` instead?"
                )

        if filled_geom is None:
            raise GGException("filled geom should not be none")

        if (
            x_scale is not None
            and not x_scale.is_empty()
            and y_scale is not None
            and not y_scale.is_empty()
        ):
            x_scale = x_scale.merge(filled_geom.gg_data.x_scale)
            y_scale = y_scale.merge(filled_geom.gg_data.y_scale)
        else:
            x_scale = filled_geom.gg_data.x_scale
            y_scale = filled_geom.gg_data.y_scale

        filled_scales.geoms.append(filled_geom)

    if x_scale is None or y_scale is None:
        raise GGException("x and y scale have not exist by this point")

    final_x_scale, _, _ = calc_tick_locations(x_scale, get_x_ticks(filled_scales))
    final_y_scale, _, _ = calc_tick_locations(y_scale, get_y_ticks(filled_scales))

    filled_scales.x_scale = final_x_scale
    filled_scales.y_scale = final_y_scale
