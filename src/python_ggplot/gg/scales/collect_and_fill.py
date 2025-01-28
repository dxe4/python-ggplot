from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import Union

from python_ggplot.colormaps.color_maps import int_to_color
from python_ggplot.core.objects import AxisKind, ColorHCL, LineType, MarkerKind, Scale
from python_ggplot.gg.datamancer_pandas_compat import (  # FormulaType,; ScalarFormula,; pandas_series_to_column,
    FormulaNode,
    GGValue,
    series_is_bool,
    series_is_float,
    series_is_int,
    series_is_obj,
    series_is_str,
    series_value_type,
)
from python_ggplot.gg.scales import (
    AlphaScale,
    GGScale,
    GGScaleDiscrete,
    ScaleValue,
    ShapeScale,
    SizeScale,
)
from python_ggplot.gg.scales.base import (
    ColorScale,
    ColorScaleKind,
    DateScale,
    FillColorScale,
    GGScaleContinuous,
    GGScaleData,
    LinearAndTransformScaleData,
    LinearDataScale,
    ScaleTransform,
    ScaleType,
    TransformedDataScale,
    scale_type_to_cls,
)
from python_ggplot.gg.scales.values import (
    AlphaScaleValue,
    ColorScaleValue,
    FillColorScaleValue,
    ShapeScaleValue,
    SizeScaleValue,
)
from python_ggplot.gg.types import DataType, DiscreteType
from python_ggplot.gg.utils import GGException


def add_identity_data(col: "str", df: pd.DataFrame, scale: GGScale):
    # TODO implement when we reach the point where is used
    raise GGException("Not implemented")


def draw_sample_idx(s_high: int, num: int = 100, seed: int = 42) -> NDArray[Any]:
    np.random.seed(seed)
    idx_num = min(num - 1, s_high)
    return np.random.randint(0, s_high + 1, size=idx_num + 1)


def is_discrete_data(
    col: pd.Series[Any],
    s: Scale,
    draw_samples: bool = True,
    discrete_threshold: float = 0.125,
) -> bool:

    if series_is_int(col):
        indices = (
            draw_sample_idx(col.high) if draw_samples else list(range(col.high + 1))
        )
        elements = {col[i] for i in indices}

        if len(elements) > round(len(indices) * discrete_threshold):
            return False
        else:
            return True

    elif series_is_float(col):
        return False

    elif series_is_str(col):
        return True

    elif series_is_bool(col):
        return True

    elif series_is_obj(col):
        indices = (
            draw_sample_idx(col.high) if draw_samples else list(range(col.high + 1))
        )
        # TODO handle pandas object type
        return False

    # TODO test this with multiple data points
    # NaT will probably fail, and there must be other cases
    raise GGException("failed to determine discreteness")


def _is_discrete(
    data: pd.Series[Any], scale: Scale, dc_kind: Optional[DiscreteType] = None
) -> bool:
    if dc_kind is None:
        return is_discrete_data(data, scale, draw_samples=True)
    return dc_kind == DiscreteType.DISCRETE


def discrete_and_type(
    data: pd.Series[Any], scale: Scale, dc_kind: Optional[DiscreteType] = None
) -> Tuple[bool, str]:
    return (_is_discrete(data, scale, dc_kind), series_value_type(data))


def fill_discrete_color_scale(
    scale_kind: ScaleType,
    value_kind: GGValue,
    column: Any,
    data_kind: DataType,
    label_seq: List[Any],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]] = None,
) -> GGScale:
    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(), label_seq=label_seq  # type ignore
    )
    if value_map_opt is not None:
        discrete_kind.value_map = value_map_opt
    else:
        color_cs = ColorHCL.gg_color_hue(len(label_seq))
        for i, k in enumerate(label_seq):
            color = color_cs[i] if data_kind == DataType.MAPPING else int_to_color(k)
            if scale_kind == ScaleType.COLOR:
                discrete_kind.value_map[k] = ColorScaleValue(color=color)
            else:
                discrete_kind.value_map[k] = FillColorScaleValue(color=color)

    cls = scale_type_to_cls(scale_kind)
    gg_data = GGScaleData(
        col=column,
        ids=set(),  # type: ignore
        value_kind=value_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )
    result = cls(gg_data=gg_data)
    return result


def fill_discrete_size_scale(
    v_kind: GGValue,
    col: FormulaNode,
    data_kind: DataType,
    label_seq: List[GGValue],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]],
    size_range: Tuple[float, float],
) -> GGScale:
    if size_range[0] != size_range[1]:
        raise GGException("Size range must be defined in this context!")

    value_map: OrderedDict[GGValue, ScaleValue] = OrderedDict()

    if value_map_opt is not None:
        value_map = value_map_opt
    else:
        num_sizes = min(len(label_seq), 5)
        min_size = size_range[0]
        max_size = size_range[1]
        step_size = (max_size - min_size) / float(num_sizes)

        for i, k in enumerate(label_seq):
            if data_kind == DataType.MAPPING:
                size = min_size + float(i) * step_size
            elif data_kind == DataType.SETTING:
                if not isinstance(k, (int, float)):
                    raise GGException("Value used to set size must be Int or Float!")
                size = float(k)
            else:
                raise GGException("unexpected data kind")

            value_map[k] = SizeScaleValue(size=size)

    discrete_kind = GGScaleDiscrete(
        value_map=value_map, label_seq=label_seq  # type ignore
    )
    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=v_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )
    result = SizeScale(
        gg_data=gg_data,
        # TODO high priority why does nim not pass  sizeRange*: tuple[low, high: float]?
        # its passed in the function, but not propagated
        # is this a bug in nim version? if so we should report it
        size_range=size_range,
        size=SizeScaleValue(size=0.0),
    )

    return result


def fill_discrete_alpha_scale(
    v_kind: GGValue,
    col: FormulaNode,
    data_kind: DataType,
    label_seq: List[GGValue],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]],
    alpha_range: Tuple[float, float],
) -> GGScale:
    # TODO refactor this
    if alpha_range[0] != alpha_range[1]:
        raise GGException("Size range must be defined in this context!")

    value_map: OrderedDict[GGValue, ScaleValue] = OrderedDict()

    if value_map_opt is not None:
        value_map = value_map_opt
    else:
        num_alphas = len(label_seq)
        min_alpha = alpha_range[0]
        max_alpha = alpha_range[1]
        step_alpha = (max_alpha - min_alpha) / float(num_alphas)

        for i, k in enumerate(label_seq):
            if data_kind == DataType.MAPPING:
                alpha = min_alpha + float(i) * step_alpha
            elif data_kind == DataType.SETTING:
                if not isinstance(k, (int, float)):
                    raise GGException("Value used to set alpha must be Int or Float!")
                alpha = float(k)
            else:
                raise GGException("unexpected data kind")

            value_map[k] = AlphaScaleValue(alpha=alpha)

    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(), label_seq=label_seq  # type ignore
    )
    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=v_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )

    # TODO high priority, this 0.0 alpha will cause bugs
    result = AlphaScale(gg_data, alpha=0.0)
    return result


def fill_discrete_shape_scale(
    v_kind: GGValue,
    col: FormulaNode,
    label_seq: List[GGValue],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]] = None,
):
    value_map: OrderedDict[GGValue, ScaleValue] = OrderedDict()
    if value_map_opt is not None:
        value_map = value_map_opt
    else:
        # TODO medium priority i dislike this modulo arithmetic here
        # re-write as a static dict so everyone can read it
        # nice and simple
        for i, k in enumerate(label_seq):
            shape = ShapeScaleValue(
                marker=MarkerKind(i % len(MarkerKind)),
                line_type=LineType((i % (len(LineType) - 1)) + 1),
            )
            value_map[k] = shape

    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(), label_seq=label_seq  # type ignore
    )
    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=v_kind,
        has_discreteness=True,
        discrete_kind=discrete_kind,
    )
    result = ShapeScale(gg_data=gg_data)
    return result


def fill_continuous_linear_scale(
    col: FormulaNode, ax_kind: AxisKind, v_kind: GGValue, data_scale: Scale
):
    discrete_kind = GGScaleContinuous(
        data_scale=data_scale,
    )

    result = LinearDataScale(
        gg_data=GGScaleData(
            col=col,
            ids=set(),  # type: ignore
            value_kind=v_kind,
            has_discreteness=True,
            discrete_kind=discrete_kind,
        ),
    )
    return result


def fill_continuous_transformed_scale(
    col: FormulaNode,
    axis_kind: AxisKind,
    value_kind: GGValue,
    trans: ScaleTransform,
    inv_trans: ScaleTransform,
    data_scale: Scale,
):

    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=value_kind,
        has_discreteness=True,
        discrete_kind=GGScaleContinuous(
            data_scale=Scale(low=trans(data_scale.low), high=trans(data_scale.high)),
        ),
    )
    result = TransformedDataScale(
        gg_data=gg_data,
        data=LinearAndTransformScaleData(
            axis_kind=axis_kind,
            reversed=False,
            transform=trans,
        ),
    )
    return result


def fill_continuous_color_scale(
    scale_type: Union[Literal[ScaleType.COLOR], Literal[ScaleType.FILL_COLOR]],
    col: FormulaNode,
    data_kind: DataType,
    value_kind: GGValue,
    data_scale: Scale,
    color_scale: ColorScale,
) -> Union[ColorScaleKind, FillColorScale]:
    cls = scale_type_to_cls(scale_type)
    if not isinstance(cls, (ColorScaleKind, FillColorScale)):
        raise GGException("expected color or fill color scale")

    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=value_kind,
        has_discreteness=False,
        discrete_kind=GGScaleContinuous(
            data_scale=data_scale,
        ),
    )

    result = cls(  # type: ignore TODO make a color mixin later
        gg_data=gg_data, color_scale=color_scale
    )

    def map_data(df: pd.DataFrame) -> List[Any]:
        # TODO FIX THIS
        result: List[Any] = []
        t_col = col.evaluate(df)
        scale_val: Dict[Any, Any] = {}
        if scale_type == ScaleType.COLOR:
            scale_val = {"kind": "color"}
        else:
            scale_val = {"kind": "fill_color"}

        if data_kind == DataType.MAPPING:
            t = t_col.to_numpy(dtype=float)
            if len(t) != len(df):
                raise GGException("Resulting array size does not match df len!")

            for val in t:
                color_idx = int(
                    round(
                        255.0
                        * ((val - data_scale.low) / (data_scale.high - data_scale.low))
                    )
                )
                color_idx = max(0, min(255, color_idx))
                c_val = color_scale.colors[color_idx]
                scale_val["color"] = int_to_color(c_val)
                result.append(scale_val.copy())
        elif data_kind == DataType.SETTING:
            t = t_col.to_numpy()
            if len(t) != len(df):
                raise GGException("Resulting array size does not match df len!")

            if np.issubdtype(t.dtype, np.integer) or np.issubdtype(t.dtype, np.str_):
                for val in t:
                    scale_val["color"] = val.to_color()
                    result.append(scale_val.copy())
            else:
                raise ValueError(
                    f"Invalid column type {t.dtype} of column {col} to set a color!"
                )
        else:
            raise GGException("expected mapping or setting for data type")
        return result

    result.map_data = map_data
    return result  # type: ignore


def fill_continuous_size_scale(
    col: FormulaNode,
    data_kind: DataType,
    value_kind: GGValue,
    data_scale: Scale,
    size_range: Tuple[float, float],
) -> SizeScale:
    """
    TODO refactor/reuse
    """
    if size_range[0] == size_range[1]:
        raise GGException("Size range must be defined in this context!")

    min_size = size_range[0]
    max_size = size_range[1]

    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=value_kind,
        has_discreteness=False,
        discrete_kind=GGScaleContinuous(
            data_scale=data_scale,
        ),
    )
    result = SizeScale(
        gg_data=gg_data,
        size=SizeScaleValue(),
        size_range=size_range,
    )

    def map_data(df: pd.DataFrame):
        result: List[Any] = []
        t = col.evaluate(df).to_numpy(dtype=float)
        if len(t) != len(df):
            raise GGException("Resulting array size does not match df len!")

        for val in t:
            if data_kind == DataType.MAPPING:
                size = (val - min_size) / (max_size - min_size)
            elif data_kind == DataType.SETTING:
                size = val
            else:
                raise GGException("incorrect mapping type")
            result.append({"kind": "size", "size": size})

        return result

    result.map_data = map_data  # type: ignore
    return result


def fill_continuous_alpha_scale(
    col: FormulaNode,
    data_kind: DataType,
    value_kind: GGValue,
    data_scale: Scale,
    alpha_range: Tuple[float, float],
) -> AlphaScale:
    """
    TODO refactor/reuse
    """
    if alpha_range[0] == alpha_range[1]:
        raise GGException("Size range must be defined in this context!")

    min_alpha = alpha_range[0]
    max_alpha = alpha_range[1]

    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=value_kind,
        has_discreteness=False,
        discrete_kind=GGScaleContinuous(
            data_scale=data_scale,
        ),
    )
    result = AlphaScale(
        gg_data=gg_data,
        alpha=0.0,  # TODO we initialise we fake value at start, after the cleanup/refactor rethink
    )

    def map_data(df: pd.DataFrame):
        t = col.evaluate(df).to_numpy(dtype=float)
        result: List[Any] = []
        alpha = 0.0
        assert len(t) == len(df), "Resulting tensor size does not match df len!"

        for val in t:
            if data_kind == DataType.MAPPING:
                alpha = (val - min_alpha) / (max_alpha - min_alpha)
            elif data_kind == DataType.SETTING:
                alpha = val
            else:
                raise GGException("incorrect mapping type")
            result.append({"kind": "alpha", "alpha": alpha})
        return result

    result.map_data = map_data  # type: ignore
    return result
