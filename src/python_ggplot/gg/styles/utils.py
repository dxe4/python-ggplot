from typing import Any, List, Tuple, cast

import pandas as pd
from typing_extensions import Dict

from python_ggplot.core.objects import GGException, Style
from python_ggplot.gg.datamancer_pandas_compat import GGValue, VNull, VString
from python_ggplot.gg.geom.base import FilledGeom, GeomType
from python_ggplot.gg.scales.base import (
    ColorScale,
    GGScale,
    GGScaleDiscrete,
    ScaleType,
    ScaleValue,
)
from python_ggplot.gg.styles.config import (
    BAR_DEFAULT_STYLE,
    HISTO_DEFAULT_STYLE,
    LINE_DEFAULT_STYLE,
    POINT_DEFAULT_STYLE,
    SMOOTH_DEFAULT_STYLE,
    TEXT_DEFAULT_STYLE,
    TILE_DEFAULT_STYLE,
)
from python_ggplot.gg.types import GGStyle, StatType

_style_lookup: Dict[GeomType, Style] = {
    GeomType.POINT: POINT_DEFAULT_STYLE,
    GeomType.BAR: BAR_DEFAULT_STYLE,
    GeomType.HISTOGRAM: HISTO_DEFAULT_STYLE,
    GeomType.TILE: TILE_DEFAULT_STYLE,
    GeomType.TEXT: TEXT_DEFAULT_STYLE,
}


def default_style(geom_type: GeomType, stat_type: StatType) -> Style:
    if geom_type == GeomType.RASTER:
        raise GGException("Warning raster does not have default style")

    if geom_type in [
        GeomType.LINE,
        GeomType.FREQ_POLY,
        GeomType.ERROR_BAR,
    ]:
        if stat_type == StatType.SMOOTH:
            return SMOOTH_DEFAULT_STYLE
        else:
            return LINE_DEFAULT_STYLE

    return _style_lookup[geom_type]


def use_or_default(c: ColorScale) -> ColorScale:
    if len(c.colors) == 0:
        return ColorScale.viridis()
    else:
        return c


def _get_field_for_user_style_merge(
    user_style: GGStyle,
    style: GGStyle,
    field_name: str,
    geom_type: GeomType,
    stat_type: StatType,
):
    user_field = getattr(user_style, field_name, None)
    style_field = getattr(style, field_name, None)
    if user_field:
        return user_field
    elif style_field:
        return style_field
    else:
        default_style_ = default_style(geom_type, stat_type)
        return getattr(default_style_, field_name, None)


def merge_user_style(style: GGStyle, fg: FilledGeom) -> Style:
    result = Style()
    if fg.gg_data.geom.gg_data.user_style is None:
        # TODO double check this logic but i think its correct
        raise GGException("User style not provided")

    u_style = fg.gg_data.geom.gg_data.user_style
    geom_type: GeomType = fg.geom_type
    stat_type = fg.gg_data.geom.gg_data.stat_kind.stat_type

    style_dict: Dict[Any, Any] = {}
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
            u_style, style, field, geom_type, stat_type
        )
        style_dict[field] = value
    result = Style(**style_dict)

    if u_style.alpha is not None:
        result.fill_color.a = u_style.alpha  # type: ignore
        if geom_type in {
            GeomType.POINT,
            GeomType.LINE,
            GeomType.ERROR_BAR,
            GeomType.TEXT,
        }:
            result.color.a = u_style.alpha
    elif style.alpha is not None:
        result.fill_color.a = style.alpha  # type: ignore
        if geom_type in {
            GeomType.POINT,
            GeomType.LINE,
            GeomType.ERROR_BAR,
            GeomType.TEXT,
        }:
            result.color.a = u_style.alpha  # type: ignore

    if result.color != result.fill_color:
        result.font.color = result.color  # type: ignore

    def_size = default_style(geom_type, stat_type).size
    if result.size != def_size:
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
    keys: List[Tuple[str, GGValue]],
):
    """
    # TODO high priority this function is unlikely to work, we will have to revisit later
    # part of doing datamancer -> pandas
    TODO Key can be str or FormulaNode. FormulaNode is from datamancer, we need to use pandas isntead
    T = Union[str, FormulaNode]
    keys: List[str, GGValue]
    """
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
                    style_val = discrete_scale.value_map[scale.gg_data.col.evaluate(df)]
                elif str(col) == str(scale.gg_data.col):
                    if (
                        isinstance(val, VNull)
                        and VString(data=str(col)) in discrete_scale.value_map
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
