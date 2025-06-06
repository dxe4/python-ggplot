from typing import Any, List, Optional, Tuple, cast

import pandas as pd

from python_ggplot.core.objects import Font, GGException, Style
from python_ggplot.gg.datamancer_pandas_compat import VNull, VString
from python_ggplot.gg.geom.base import Geom, GeomType
from python_ggplot.gg.geom.filled_geom import FilledGeom
from python_ggplot.gg.scales.base import (
    ColorScale,
    GGScale,
    GGScaleDiscrete,
    ScaleType,
    ScaleValue,
)
from python_ggplot.gg.types import GGStyle


def use_or_default(c: Optional[ColorScale]) -> ColorScale:
    if c is None or len(c.colors) == 0:
        return ColorScale.viridis()
    else:
        return c


def _get_field_for_user_style_merge(
    user_style: GGStyle, style: GGStyle, field_name: str, geom: Geom
):
    user_field = getattr(user_style, field_name, None)
    style_field = getattr(style, field_name, None)
    if user_field:
        return user_field
    elif style_field:
        return style_field
    else:

        default_style_ = geom.default_style()
        return getattr(default_style_, field_name, None)


def merge_user_style(style: GGStyle, fg: FilledGeom) -> Style:

    geom_type: GeomType = fg.geom_type
    result = fg.gg_data.geom.default_style()

    if fg.gg_data.geom.gg_data.user_style is None:
        # TODO double check this logic but i think its correct
        raise GGException("User style not provided")

    user_style = fg.gg_data.geom.gg_data.user_style

    for field in [
        "color",
        "size",
        "line_type",
        "line_width",
        "fill_color",
        "marker",
        "error_bar_kind",
        "font",
    ]:
        value = _get_field_for_user_style_merge(
            user_style, style, field, fg.gg_data.geom
        )
        if value is not None:
            setattr(result, field, value)

    if user_style.alpha is not None:
        result.fill_color = result.fill_color.update_with_copy(a=user_style.alpha)

        if geom_type in {
            GeomType.POINT,
            GeomType.LINE,
            GeomType.ERROR_BAR,
            GeomType.TEXT,
        }:
            result.color = result.color.update_with_copy(a=user_style.alpha)
    elif style.alpha is not None:
        result.fill_color = result.fill_color.update_with_copy(a=style.alpha)
        if geom_type in {
            GeomType.POINT,
            GeomType.LINE,
            GeomType.ERROR_BAR,
            GeomType.TEXT,
        }:
            result.color = result.color.update_with_copy(a=style.alpha)

    # TODO check why this is None? Should it be None or WTF?
    if result.font is None:
        result.font = Font()

    if result.color != result.fill_color:
        result.font.color = result.color  # type: ignore

    default_size = fg.gg_data.geom.default_style().size
    if result.size != default_size:
        result.font.size = result.size * 2.5  # type: ignore

    return result


def change_style(style: GGStyle, scale_value: ScaleValue) -> GGStyle:

    try:
        scale_value.update_style(style)
        return style
    # TODO this needs refactoring
    except NotImplementedError:
        raise GGException(
            f"Setting style of {scale_value.scale_type} not supported at the moment!"
        )


def apply_style(
    style: GGStyle,
    df: pd.DataFrame,
    scales: List[GGScale],
    keys: List[Tuple[str, Any]],
):
    for col, val in keys:
        for scale in scales:
            if scale.scale_type in {
                ScaleType.LINEAR_DATA,
                ScaleType.TRANSFORMED_DATA,
                ScaleType.TEXT,
            }:
                continue

            if scale.is_discrete():
                is_col = col in df.columns
                discrete_scale = cast(GGScaleDiscrete, scale.gg_data.discrete_kind)
                if not is_col:
                    col_vals = scale.gg_data.col.evaluate(df)
                    if len(col_vals.unique()) > 1:
                        raise GGException("Expected one unique value")
                    value = col_vals.iloc[0]
                    # TODO value map is sometypes not obeying the type definition of GGValue
                    # this can cause omse bugs, better to fix it asap
                    style_val = discrete_scale.value_map[value]
                elif str(col) == str(scale.gg_data.col):
                    # TODO the or part is temporary
                    if isinstance(val, VNull) and (
                        VString(data=str(col)) in discrete_scale.value_map
                        or col in discrete_scale.value_map
                    ):
                        style_val = discrete_scale.value_map[VString(data=str(col))]
                    elif val in discrete_scale.value_map:
                        style_val = discrete_scale.value_map[val]
                    else:
                        continue
                else:
                    continue

                style = change_style(style, style_val)
    return style
