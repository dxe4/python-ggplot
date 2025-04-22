from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_ggplot.common.maths import histogram
from python_ggplot.core.objects import GGException, Scale
from python_ggplot.gg.datamancer_pandas_compat import VNull, VString
from python_ggplot.gg.geom.base import GeomType, HistogramDrawingStyle
from python_ggplot.gg.types import (
    COUNT_COL,
    PREV_VALS_COL,
    SMOOTH_VALS_COL,
    BinByType,
    GGStyle,
    PositionType,
    SmoothMethodType,
    StatBin,
    StatSmooth,
)

if TYPE_CHECKING:
    from python_ggplot.gg.geom.base import FilledGeom, FilledStatGeom, Geom
    from python_ggplot.gg.scales.base import FilledScales, GGScale


def add_zero_keys(
    df: pd.DataFrame, keys: pd.Series, x_col: Any, count_col: str
) -> pd.DataFrame:
    exist_keys = df[x_col].unique()  # type: ignore
    df_zero = pd.DataFrame({x_col: keys[~keys.isin(exist_keys)]})  # type: ignore
    df_zero[count_col] = 0
    return pd.concat([df, df_zero], ignore_index=True)


def call_smoother(
    fg: "FilledGeom", df: pd.DataFrame, scale: "GGScale", range: Any
) -> NDArray[np.floating[Any]]:

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


def call_hist(
    df: pd.DataFrame,
    bins_arg: Any,
    stat_kind: StatBin,
    range_scale: Scale,
    weight_scale: Optional["GGScale"],
    data: NDArray[np.floating[Any]],
):
    if stat_kind.bin_by == BinByType.FULL:
        range_val = (range_scale.low, range_scale.high)
    else:
        range_val = (0.0, 0.0)

    weight_data = _scale_to_numpy_array(df, weight_scale)
    if len(weight_data) == 0:
        weight_data = None

    hist, bin_edges = histogram(
        data,
        bins_arg,
        weights=weight_data,
        range=range_val,
        density=stat_kind.density,
    )
    return hist, bin_edges


def _scale_to_numpy_array(
    df: pd.DataFrame, scale: Optional["GGScale"]
) -> NDArray[np.floating[Any]]:
    if scale is None:
        return np.empty(0, dtype=np.float64)
    else:
        return df[str(scale.gg_data.col)].to_numpy(dtype=float)  # type: ignore


def call_histogram(
    geom: "Geom",
    df: pd.DataFrame,
    scale: Optional["GGScale"],
    weight_scale: Optional["GGScale"],
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

    data = _scale_to_numpy_array(df, scale)
    hist = []
    bin_edges = []
    bin_widths = []

    if stat_kind.bin_edges is not None:
        hist, bin_edges = call_hist(
            df, stat_kind.bin_edges, stat_kind, range_scale, weight_scale, data
        )
    elif stat_kind.bin_width is not None:
        bins = round((range_scale.high - range_scale.low) / stat_kind.bin_width)
        hist, bin_edges = call_hist(
            df, int(bins), stat_kind, range_scale, weight_scale, data
        )
    else:
        hist, bin_edges = call_hist(
            df, stat_kind.num_bins, stat_kind, range_scale, weight_scale, data
        )

    bin_widths = np.diff(bin_edges)  # type: ignore
    # TODO CRITICAL+ sanity this logic
    # those go in a df, they have to be of the same size, but clearly  the diff will be off by 1
    # sanity logic, and probably use the builint hist for this
    bin_widths = np.concatenate(([0.0], bin_widths))
    hist = np.append(hist, 0.0)  # type: ignore
    return hist, bin_edges, bin_widths  # type: ignore


def count_(
    df: pd.DataFrame,  # type: ignore
    x_col: str,
    name: str,
    weights: Optional["GGScale"] = None,
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


def apply_cont_scale_if_any(
    yield_df: pd.DataFrame,
    scales: List["GGScale"],
    base_style: GGStyle,
    geom_type: GeomType,
    to_clone: bool = False,
):
    from python_ggplot.gg.scales.base import ScaleType
    from python_ggplot.gg.styles.utils import change_style

    result_style = base_style
    result_styles = []
    result_df = yield_df.copy() if to_clone else yield_df

    for scale in scales:
        # TODO col eval is a global issue, fine for now
        result_df[scale.get_col_name()] = scale.gg_data.col.evaluate(result_df)  # type: ignore

        if scale.scale_type in {ScaleType.TRANSFORMED_DATA, ScaleType.LINEAR_DATA}:
            pass
        else:
            # avoid expensive computation for raster
            if geom_type != GeomType.RASTER:
                # TODO high priority map_data logic is funny overall, add ignore type for now
                print(scale)
                print(scale.map_data)
                sc_vals = scale.map_data(result_df)
                result_styles = [change_style(base_style, val) for val in sc_vals]

    if not result_styles:
        result_styles = [base_style]

    return (result_style, result_styles, result_df)


def add_counts_by_position(
    col_sum: pd.Series,  # type: ignore
    col: pd.Series,  # type: ignore
    pos: Optional[PositionType],
) -> pd.Series:
    # TODO use is_numeric_dtype in other places of the code base
    if pd.api.types.is_numeric_dtype(col):  # type: ignore
        if pos == PositionType.STACK:
            if len(col_sum) == 0:
                return col.copy()
            else:
                return col_sum + col
        elif pos in (PositionType.IDENTITY, PositionType.DODGE):
            return col.copy()
        elif pos == PositionType.FILL:
            return pd.Series([1.0])
        else:
            raise GGException("unexpected position type")
    else:
        return col.copy()


def _modify_for_stacking(geom: "Geom") -> bool:
    if geom.gg_data.position != PositionType.STACK:
        return False
    is_bar = (
        geom.geom_type == GeomType.HISTOGRAM
        and geom.gg_data.histogram_drawing_style == HistogramDrawingStyle.BARS
    ) or (geom.geom_type == GeomType.BAR)
    if not is_bar:
        return True
    return False


def _filled_identity_geom_map(
    df: pd.DataFrame,
    filled_scales: "FilledScales",
    filled_stat_geom: "FilledStatGeom",
    filled_geom: "FilledGeom",
    style: "GGStyle",
) -> "FilledGeom":
    from python_ggplot.gg.styles.utils import apply_style

    geom = filled_stat_geom.geom
    grouped = df.groupby(filled_stat_geom.map_discrete_columns, sort=True)  # type: ignore
    sorted_keys = sorted(grouped.groups.keys(), reverse=True)  # type: ignore
    col = pd.Series(dtype=float)  # type: ignore

    for keys in sorted_keys:  # type: ignore
        sub_df = grouped.get_group(keys)  # type: ignore
        key_values = list(product(filled_stat_geom.map_discrete_columns, [keys]))  # type: ignore
        current_style = apply_style(
            deepcopy(style), sub_df, filled_stat_geom.discrete_scales, key_values
        )  # type: ignore

        yield_df: pd.DataFrame = sub_df.copy()  # type: ignore
        filled_stat_geom.x.set_x_attributes(filled_geom, yield_df)

        if geom.gg_data.position == PositionType.STACK:
            yield_df[PREV_VALS_COL] = 0.0 if len(col) == 0 else col.copy()  # type: ignore

        col = add_counts_by_position(
            yield_df[filled_geom.gg_data.y_col],  # type: ignore
            col,  # type: ignore
            geom.gg_data.position,
        )

        if _modify_for_stacking(geom):
            yield_df[filled_geom.gg_data.y_col] = col

        yield_df = filled_geom.maybe_filter_unique(yield_df)
        # this has to be copied otherwise the same style is changed
        base_style = deepcopy(current_style)
        style_, styles_, temp_yield_df = apply_cont_scale_if_any(
            yield_df,
            filled_stat_geom.continuous_scales,
            base_style,
            geom.geom_type,
            to_clone=True,
        )
        filled_geom.gg_data.yield_data[keys] = (style_, styles_, temp_yield_df)  # type: ignore

    if geom.gg_data.position == PositionType.STACK and filled_geom.is_discrete_y():
        filled_geom.gg_data.y_scale = filled_geom.gg_data.y_scale.merge(
            Scale(low=filled_geom.gg_data.y_scale.low, high=col.max())  # type: ignore
        )

    if (
        geom.geom_type == GeomType.HISTOGRAM
        and geom.gg_data.position == PositionType.STACK
        and geom.gg_data.histogram_drawing_style == HistogramDrawingStyle.OUTLINE
    ):
        filled_geom.gg_data.yield_data = dict(reversed(list(filled_geom.gg_data.yield_data.items())))  # type: ignore

    return filled_geom


def _filled_identity_geom_set(
    df: pd.DataFrame,
    filled_scales: "FilledScales",
    filled_stat_geom: "FilledStatGeom",
    filled_geom: "FilledGeom",
    style: "GGStyle",
) -> "FilledGeom":
    yield_df: pd.DataFrame = df.copy()
    yield_df[PREV_VALS_COL] = 0.0
    yield_df = filled_geom.maybe_filter_unique(yield_df)
    filled_stat_geom.x.set_x_attributes(filled_geom, yield_df)
    key = ("", None)
    filled_geom.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
        yield_df,
        filled_stat_geom.continuous_scales,
        style,
        filled_stat_geom.geom.geom_type,
    )
    return filled_geom


def _filled_count_geom_map(
    df: pd.DataFrame,
    filled_scales: "FilledScales",
    filled_stat_geom: "FilledStatGeom",
    filled_geom: "FilledGeom",
    style: "GGStyle",
) -> "FilledGeom":
    from python_ggplot.gg.styles.utils import apply_style

    grouped = df.groupby(filled_stat_geom.map_discrete_columns, sort=False)  # type: ignore
    sorted_keys = sorted(grouped.groups.keys(), reverse=True)  # type: ignore

    # TODO fix col type, issue with pandas index
    col = pd.Series(dtype=float)  # For stacking

    all_classes = pd.Series(df[filled_stat_geom.get_x_col()].unique())  # type: ignore
    if len(filled_stat_geom.continuous_scales) > 0:
        raise GGException("continuous_scales > 0")

    for keys in sorted_keys:  # type: ignore
        sub_df = grouped.get_group(keys)  # type: ignore
        key_values = list(product(filled_stat_geom.map_discrete_columns, [keys]))  # type: ignore
        current_style = apply_style(
            deepcopy(style), sub_df, filled_stat_geom.discrete_scales, key_values
        )  # type: ignore

        weight_scale = filled_scales.get_weight_scale(
            filled_stat_geom.geom, optional=True
        )
        yield_df = count_(sub_df, filled_stat_geom.get_x_col(), "", weight_scale)  # type: ignore

        yield_df = add_zero_keys(yield_df, all_classes, filled_stat_geom.get_x_col(), "count")  # type: ignore
        yield_df = yield_df.sort_values(filled_stat_geom.get_x_col())  # type: ignore
        yield_df = yield_df.reset_index(drop=True)

        if filled_stat_geom.geom.gg_data.position == PositionType.STACK:
            yield_df["prev_vals"] = 0.0 if len(col) == 0 else col.copy()

        col = add_counts_by_position(
            col, yield_df["count"], filled_stat_geom.geom.gg_data.position  # type: ignore
        )
        col = col.to_numpy()

        if _modify_for_stacking(filled_stat_geom.geom):
            y_col = filled_stat_geom.get_y_col()
            yield_df[y_col] = col

        yield_df = filled_geom.maybe_filter_unique(yield_df)

        filled_geom.gg_data.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
            yield_df,
            filled_stat_geom.continuous_scales,
            current_style,
            filled_stat_geom.geom.geom_type,
            to_clone=True,
        )

        filled_stat_geom.x.set_x_attributes(filled_geom, yield_df)

        filled_geom.gg_data.y_scale = filled_geom.gg_data.y_scale.merge(
            Scale(low=0.0, high=float(col.max()))  # type: ignore
        )
    return filled_geom


def _filled_count_geom_set(
    df: pd.DataFrame,
    filled_scales: "FilledScales",
    filled_stat_geom: "FilledStatGeom",
    filled_geom: "FilledGeom",
    style: "GGStyle",
) -> "FilledGeom":
    weight_scale = filled_scales.get_weight_scale(filled_stat_geom.geom, optional=True)
    yield_df = count_(df, filled_stat_geom.get_x_col(), COUNT_COL, weight_scale)
    # TODO double check prev_vals
    yield_df[PREV_VALS_COL] = 0.0

    key = ("", VNull())
    yield_df = filled_geom.maybe_filter_unique(yield_df)
    filled_geom.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
        yield_df,
        filled_stat_geom.continuous_scales,
        style,
        filled_stat_geom.geom.geom_type,
    )

    filled_stat_geom.x.set_x_attributes(filled_geom, yield_df)
    filled_geom.gg_data.y_scale = filled_geom.gg_data.y_scale.merge(
        Scale(low=0.0, high=float(yield_df[COUNT_COL].max()))  # type: ignore
    )
    return filled_geom


def _filled_bin_geom_map(
    df: pd.DataFrame,
    filled_scales: "FilledScales",
    filled_stat_geom: "FilledStatGeom",
    filled_geom: "FilledGeom",
    style: "GGStyle",
) -> "FilledGeom":
    DISABLE_MODIFY_FOR_STACKING = True
    from python_ggplot.gg.styles.utils import apply_style

    grouped = df.groupby(filled_stat_geom.map_discrete_columns, sort=True)  # type: ignore TODO

    sorted_keys = sorted(grouped.groups.keys(), reverse=True)  # type: ignore
    col = pd.Series(dtype=float)

    for keys in sorted_keys:  # type: ignore
        sub_df = grouped.get_group(keys)  # type: ignore
        key_values = list(product(filled_stat_geom.map_discrete_columns, [keys]))  # type: ignore
        current_style = apply_style(
            deepcopy(style), sub_df, filled_stat_geom.discrete_scales, key_values
        )  # type: ignore
        hist, bins, _ = call_histogram(
            filled_stat_geom.geom,
            sub_df,  # type: ignore
            filled_stat_geom.x,
            filled_scales.get_weight_scale(filled_stat_geom.geom, optional=True),
            filled_stat_geom.x.gg_data.discrete_kind.data_scale,  # type: ignore TODO
        )

        count_col = filled_stat_geom.count_col()  # type: ignore
        yield_df = pd.DataFrame({
            filled_stat_geom.x.get_col_name(): bins,
            count_col: hist
        })

        if filled_stat_geom.geom.gg_data.position == PositionType.STACK:
            yield_df[PREV_VALS_COL] = col if len(col) > 0 else 0.0

        col = add_counts_by_position(
            col, hist, filled_stat_geom.geom.gg_data.position
        )

        # TODO CRITICAL+
        # This does what was intented, it adds the previous values to the current
        # if previous his is 10,10 and current is 0,1, we end up with 10, 11
        # the issue is this eventually draws 10 + 10 instead of 10 + 0
        # revisit this later, for now disable
        if _modify_for_stacking(filled_stat_geom.geom) and not DISABLE_MODIFY_FOR_STACKING:
            yield_df[filled_geom.gg_data.y_col] = col

        yield_df = filled_geom.maybe_filter_unique(yield_df)
        filled_geom.gg_data.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
            yield_df,
            filled_stat_geom.continuous_scales,
            current_style,
            filled_stat_geom.geom.geom_type,
            to_clone=True,  # type: ignore
        )

        filled_geom.gg_data.num_x = max(filled_geom.gg_data.num_x, len(yield_df))

        if filled_stat_geom.geom.geom_type == GeomType.FREQ_POLY:
            bin_width = float(bins[1] - bins[0]) if len(bins) > 1 else 0.0
            filled_geom.gg_data.x_scale = filled_geom.gg_data.x_scale.merge(
                Scale(
                    low=float(min(bins)) - bin_width / 2.0,
                    high=float(max(bins)) + bin_width / 2.0,
                )
            )
        else:
            filled_geom.gg_data.x_scale = filled_geom.gg_data.x_scale.merge(
                Scale(low=float(min(bins)), high=float(max(bins)))
            )

        filled_geom.gg_data.y_scale = filled_geom.gg_data.y_scale.merge(
            Scale(low=0.0, high=float(col.max()))  # type: ignore
        )
    return filled_geom


def _filled_bin_geom_set(
    df: pd.DataFrame,
    filled_scales: "FilledScales",
    filled_stat_geom: "FilledStatGeom",
    filled_geom: "FilledGeom",
    style: "GGStyle",
) -> "FilledGeom":
    hist, bins, bin_widths = call_histogram(
        filled_stat_geom.geom,
        df,
        filled_stat_geom.x,
        filled_scales.get_weight_scale(filled_stat_geom.geom, optional=True),
        filled_stat_geom.x.gg_data.discrete_kind.data_scale,  # type: ignore TODO
    )

    count_col = filled_stat_geom.count_col()  # type: ignore
    width_col = filled_stat_geom.width_col()  # type: ignore
    yield_df = pd.DataFrame(
        {
            filled_stat_geom.x.get_col_name(): bins,
            count_col: hist,
            width_col: bin_widths,
        }
    )
    yield_df[PREV_VALS_COL] = 0.0
    yield_df = filled_geom.maybe_filter_unique(yield_df)

    key = ("", VNull())

    if len(filled_stat_geom.continuous_scales) != 0:
        raise GGException("seems the data is discrete")

    filled_geom.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
        yield_df,
        filled_stat_geom.continuous_scales,
        style,
        filled_stat_geom.geom.geom_type,
    )
    filled_geom.gg_data.num_x = len(yield_df)
    filled_geom.gg_data.x_scale = filled_geom.gg_data.x_scale.merge(
        Scale(low=float(min(bins)), high=float(max(bins)))
    )

    filled_geom.gg_data.y_scale = filled_geom.gg_data.y_scale.merge(
        Scale(low=0.0, high=float(max(hist)))
    )
    return filled_geom


def _filled_smooth_geom_map(
    df: pd.DataFrame,
    filled_scales: "FilledScales",
    filled_stat_geom: "FilledStatGeom",
    filled_geom: "FilledGeom",
    style: "GGStyle",
) -> "FilledGeom":
    from python_ggplot.gg.styles.utils import apply_style

    grouped = df.groupby(filled_stat_geom.map_discrete_columns, sort=True)  # type: ignore
    sorted_keys = sorted(grouped.groups.keys(), reverse=True)  # type: ignore
    col = pd.Series(dtype=float)  # type: ignore

    for keys in sorted_keys:  # type: ignore
        sub_df = grouped.get_group(keys)  # type: ignore
        key_values = list(product(filled_stat_geom.map_discrete_columns, [keys]))  # type: ignore
        current_style = apply_style(
            deepcopy(style), sub_df, filled_stat_geom.discrete_scales, key_values
        )  # type: ignore

        yield_df = sub_df.copy()  # type: ignore

        smoothed = call_smoother(
            filled_geom,
            yield_df,  # type: ignore
            filled_stat_geom.y,
            # This has to be continuous for data scale to exist needs cleanup
            range=filled_stat_geom.x.gg_data.discrete_kind.data_scale,  # type: ignore
        )
        yield_df[SMOOTH_VALS_COL] = smoothed

        filled_stat_geom.x.set_x_attributes(filled_geom, yield_df)  # type: iignore

        if filled_stat_geom.geom.gg_data.position == PositionType.STACK:
            yield_df[PREV_VALS_COL] = pd.Series(0.0, index=yield_df.index) if len(col) == 0 else col.copy()  # type: ignore

        # possibly modify `col` if stacking
        # TODO double check this
        yield_df[filled_geom.gg_data.y_col] = add_counts_by_position(
            yield_df[filled_geom.gg_data.y_col],  # type: ignore
            col,  # type: ignore
            filled_stat_geom.geom.gg_data.position,
        )

        if _modify_for_stacking(filled_stat_geom.geom):
            yield_df[result.y_col] = col  # type: ignore

        yield_df = filled_geom.maybe_filter_unique(yield_df)
        filled_geom.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, current_style, geom.geom_type, to_clone=True  # type: ignore
        )

    if (
        filled_stat_geom.geom.gg_data.position == PositionType.STACK
        and not filled_geom.is_discrete_y()
    ):
        # only update required if stacking, as we've computed the range beforehand
        filled_geom.gg_data.y_scale = filled_geom.gg_data.y_scale.merge(
            Scale(low=result.gg_data.y_scale.low, high=col.max())  # type: ignore
        )

    if (
        filled_stat_geom.geom.geom_type == GeomType.HISTOGRAM
        and filled_stat_geom.geom.gg_data.position == PositionType.STACK
        and filled_stat_geom.geom.gg_data.histogram_drawing_style
        == HistogramDrawingStyle.OUTLINE
    ):
        filled_geom.gg_data.yield_data = dict(reversed(list(filled_geom.gg_data.yield_data.items())))  # type: ignore

    return filled_geom


def _filled_smooth_geom_set(
    df: pd.DataFrame,
    filled_scales: "FilledScales",
    filled_stat_geom: "FilledStatGeom",
    filled_geom: "FilledGeom",
    style: "GGStyle",
) -> "FilledGeom":
    yield_df = df.copy()
    smoothed = call_smoother(
        filled_geom, yield_df, filled_stat_geom.y, range=filled_stat_geom.x.data_scale  # type: ignore TODO critical FIX
    )
    yield_df[PREV_VALS_COL] = pd.Series(0.0, index=yield_df.index)  # type: ignore
    yield_df[SMOOTH_VALS_COL] = smoothed
    yield_df = filled_geom.maybe_filter_unique(yield_df)

    filled_stat_geom.x.set_x_attributes(filled_geom, yield_df)

    key = ("", VNull())
    filled_geom.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
        yield_df,
        filled_stat_geom.continuous_scales,
        style,
        filled_stat_geom.geom.geom_type,
    )
    return filled_geom


def filled_identity_geom(
    df: pd.DataFrame,
    filled_geom: "FilledGeom",
    filled_stat_geom: "FilledStatGeom",
    filled_scales: "FilledScales",
    style: "GGStyle",
) -> "FilledGeom":

    if len(filled_stat_geom.map_discrete_columns) > 0:
        filled_geom = _filled_identity_geom_map(
            df, filled_scales, filled_stat_geom, filled_geom, style
        )
    else:
        filled_geom = _filled_identity_geom_set(
            df, filled_scales, filled_stat_geom, filled_geom, style
        )

    if filled_stat_geom.y is not None and filled_stat_geom.y.is_discrete():
        # TODO fix
        # y.label_seqwill exist since is discrete, but this needs refactor anyway
        filled_geom.gg_data.y_discrete_kind.label_seq = filled_stat_geom.y.gg_data.discrete_kind.label_seq  # type: ignore

    filled_geom.gg_data.num_y = filled_geom.gg_data.num_x
    return filled_geom


def filled_count_geom(
    df: pd.DataFrame,
    filled_geom: "FilledGeom",
    filled_stat_geom: "FilledStatGeom",
    filled_scales: "FilledScales",
    style: "GGStyle",
) -> "FilledGeom":
    all_classes = df[filled_stat_geom.get_x_col()].unique()  # type: ignore

    if len(filled_stat_geom.map_discrete_columns) > 0:
        filled_geom = _filled_count_geom_map(
            df, filled_scales, filled_stat_geom, filled_geom, style
        )
    else:
        if len(filled_stat_geom.continuous_scales) > 0:
            raise GGException("continuous_scales > 0")
        filled_geom = _filled_count_geom_set(
            df, filled_scales, filled_stat_geom, filled_geom, style
        )

    filled_geom.gg_data.num_y = round(filled_geom.gg_data.y_scale.high)
    filled_geom.gg_data.num_x = len(all_classes)  # type: ignore

    if filled_geom.gg_data.num_x != len(all_classes):  # type: ignore
        # todo provide better messages...
        raise GGException("ERROR")

    return filled_geom


def filled_bin_geom(
    df: pd.DataFrame,
    filled_geom: "FilledGeom",
    filled_stat_geom: "FilledStatGeom",
    filled_scales: "FilledScales",
    style: "GGStyle",
) -> "FilledGeom":

    if len(filled_stat_geom.map_discrete_columns) > 0:
        filled_geom = _filled_bin_geom_map(
            df, filled_scales, filled_stat_geom, filled_geom, style
        )
    else:
        filled_geom = _filled_bin_geom_set(
            df, filled_scales, filled_stat_geom, filled_geom, style
        )

    filled_geom.gg_data.num_y = round(filled_geom.gg_data.y_scale.high)

    if filled_stat_geom.x.is_discrete():
        # TODO fix, this is an error
        filled_geom.gg_data.x_label_seq = filled_stat_geom.x.gg_data.label_seq  # type: ignore

    return filled_geom


def filled_smooth_geom(
    df: pd.DataFrame,
    filled_geom: "FilledGeom",
    filled_stat_geom: "FilledStatGeom",
    filled_scales: "FilledScales",
    style: "GGStyle",
) -> "FilledGeom":
    if len(filled_stat_geom.map_discrete_columns) > 0:
        filled_geom = _filled_smooth_geom_map(
            df, filled_scales, filled_stat_geom, filled_geom, style
        )
    else:
        filled_geom = _filled_smooth_geom_set(
            df, filled_scales, filled_stat_geom, filled_geom, style
        )

    filled_geom.gg_data.num_y = filled_geom.gg_data.num_x

    return filled_geom
