from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_ggplot.core.chroma import int_to_color
from python_ggplot.core.objects import AxisKind, ColorHCL, LineType, MarkerKind, Scale
from python_ggplot.gg.datamancer_pandas_compat import (
    VTODO,
    GGValue,
    VectorCol,
    VNull,
    VString,
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
    AbstractGGScale,
    AlphaScaleValue,
    ColorScale,
    ColorScaleKind,
    ColorScaleValue,
    FillColorScale,
    FillColorScaleValue,
    FilledScales,
    GGScaleContinuous,
    GGScaleData,
    LinearAndTransformScaleData,
    LinearDataScale,
    MainAddScales,
    ScaleTransformFunc,
    ScaleType,
    ShapeScaleValue,
    SizeScaleValue,
    TextScale,
    TransformedDataScale,
    scale_type_to_cls,
)
from python_ggplot.gg.scales.post_process import post_process_scales
from python_ggplot.gg.styles.config import (
    DEFAULT_ALPHA_RANGE_TUPLE,
    DEFAULT_SIZE_RANGE_TUPLE,
)
from python_ggplot.gg.types import DataType, DiscreteType, GgPlot
from python_ggplot.gg.utils import GGException


def add_identity_data(col: "str", df: pd.DataFrame, scale: GGScale):
    # TODO implement when we reach the point where is used
    raise GGException("Not implemented")


def draw_sample_idx(s_high: int, num: int = 100, seed: int = 42) -> NDArray[Any]:
    np.random.seed(seed)
    idx_num = min(num - 1, s_high)
    return np.random.randint(0, s_high + 1, size=idx_num + 1)


def is_discrete_data(
    col: pd.Series,  # type: ignore
    scale: GGScale,
    draw_samples: bool = True,
    discrete_threshold: float = 0.125,
) -> bool:
    """
    TODO: high priority
    sanity check this logic
    seems scale was only used for logging and exception messages
    need to double check it
    """

    if series_is_int(col):
        indices = (
            draw_sample_idx(col.max())  # type: ignore
            if draw_samples
            else list(range(col.max() + 1))  # type: ignore
        )
        elements = {col[i] for i in indices}  # type: ignore

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
    data: pd.Series,  # type: ignore
    scale: GGScale,
    dc_kind: Optional[DiscreteType] = None,
) -> bool:
    if dc_kind is None:
        return is_discrete_data(data, scale, draw_samples=True)
    return dc_kind == DiscreteType.DISCRETE


def discrete_and_type(
    data: pd.Series,  # type: ignore
    scale: GGScale,
    dc_kind: Optional[DiscreteType] = None,
) -> Tuple[bool, str]:
    return (_is_discrete(data, scale, dc_kind), series_value_type(data))


def fill_discrete_color_scale(
    scale_kind: ScaleType,
    value_kind: GGValue,
    column: Any,
    data_type: DataType,
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
            color = color_cs[i] if data_type == DataType.MAPPING else int_to_color(k)
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
        data_type=data_type,
        discrete_kind=discrete_kind,
    )
    result = cls(gg_data=gg_data)
    return result


def fill_discrete_size_scale(
    value_kind: GGValue,
    col: VectorCol,
    data_type: DataType,
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
            if data_type == DataType.MAPPING:
                size = min_size + float(i) * step_size
            elif data_type == DataType.SETTING:
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
        value_kind=value_kind,
        has_discreteness=True,
        data_type=data_type,
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
    value_kind: GGValue,
    col: VectorCol,
    data_type: DataType,
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
            if data_type == DataType.MAPPING:
                alpha = min_alpha + float(i) * step_alpha
            elif data_type == DataType.SETTING:
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
        value_kind=value_kind,
        has_discreteness=True,
        data_type=data_type,
        discrete_kind=discrete_kind,
    )

    # TODO high priority, this 0.0 alpha will cause bugs
    result = AlphaScale(gg_data, alpha=0.0)
    return result


def fill_discrete_shape_scale(
    value_kind: GGValue,
    col: VectorCol,
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
        value_kind=value_kind,
        has_discreteness=True,
        discrete_kind=discrete_kind,
    )
    result = ShapeScale(gg_data=gg_data)
    return result


def fill_discrete_linear_trans_scale(
    scale_type: ScaleType,
    col: VectorCol,
    ax_kind: AxisKind,
    value_kind: GGValue,
    label_seq: List[GGValue],
    trans: Optional[ScaleTransformFunc] = None,
    inv_trans: Optional[ScaleTransformFunc] = None,
) -> GGScale:
    cls = scale_type_to_cls(scale_type)
    if not cls in (TransformedDataScale, LinearDataScale):
        raise GGException(f"expected transform or linear received {cls.__name__}")

    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=value_kind,
        has_discreteness=True,
        discrete_kind=GGScaleDiscrete(
            value_map=OrderedDict(), label_seq=label_seq  # type ignore
        ),
    )
    # TODO high priority, we need to set the axis kind
    # the original version is a bit confusing,
    # there's a bunch of required attrs that arent set
    # result.ax_kind = ax_kind
    # need to look further
    result = cls(gg_data=gg_data)  # type: ignore

    if scale_type == ScaleType.TRANSFORMED_DATA:
        if trans is None:
            raise ValueError("trans must not be None when sc_kind is TRANSFORMED_DATA")
        # TODO this needs double checking later
        result.trans = trans
        result.inv_trans = inv_trans

    return result  # type: ignore


def fill_continuous_linear_scale(
    col: VectorCol, ax_kind: AxisKind, value_kind: GGValue, data_scale: Scale
):
    discrete_kind = GGScaleContinuous(
        data_scale=data_scale,
    )

    result = LinearDataScale(
        gg_data=GGScaleData(
            col=col,
            ids=set(),  # type: ignore
            value_kind=value_kind,
            has_discreteness=True,
            discrete_kind=discrete_kind,
        ),
    )
    return result


def fill_continuous_transformed_scale(
    col: VectorCol,
    axis_kind: AxisKind,
    value_kind: GGValue,
    trans: ScaleTransformFunc,
    inv_trans: ScaleTransformFunc,
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
    col: VectorCol,
    data_type: DataType,
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

        if data_type == DataType.MAPPING:
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
        elif data_type == DataType.SETTING:
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
    col: VectorCol,
    data_type: DataType,
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
            if data_type == DataType.MAPPING:
                size = (val - min_size) / (max_size - min_size)
            elif data_type == DataType.SETTING:
                size = val
            else:
                raise GGException("incorrect mapping type")
            result.append({"kind": "size", "size": size})

        return result

    result.map_data = map_data  # type: ignore
    return result


def fill_continuous_alpha_scale(
    col: VectorCol,
    data_type: DataType,
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
            if data_type == DataType.MAPPING:
                alpha = (val - min_alpha) / (max_alpha - min_alpha)
            elif data_type == DataType.SETTING:
                alpha = val
            else:
                raise GGException("incorrect mapping type")
            result.append({"kind": "alpha", "alpha": alpha})
        return result

    result.map_data = map_data  # type: ignore
    return result


def fill_scale_impl(
    value_kind: GGValue,
    is_discrete: bool,
    col: VectorCol,
    scale_type: ScaleType,
    data_type: DataType,
    label_seq: Optional[List[GGValue]] = None,
    value_map: Optional[OrderedDict[GGValue, ScaleValue]] = None,
    data_scale: Optional[Scale] = None,
    ax_kind: Optional[AxisKind] = None,
    trans: Optional[ScaleTransformFunc] = None,
    inv_trans: Optional[ScaleTransformFunc] = None,
    color_scale: Optional[ColorScale] = None,
    size_range: Tuple[float, float] = DEFAULT_SIZE_RANGE_TUPLE,
    alpha_range: Tuple[float, float] = DEFAULT_ALPHA_RANGE_TUPLE,
) -> GGScale:
    """
    TODO refactor ASAP
    this is a mess
    """
    if color_scale is None:
        color_scale = ColorScale.viridis()

    if is_discrete:
        if label_seq is not None:
            labels = label_seq
        else:
            labels = []

        if scale_type == ScaleType.COLOR:
            return fill_discrete_color_scale(
                ScaleType.COLOR, value_kind, col, data_type, labels, value_map
            )
        elif scale_type == ScaleType.FILL_COLOR:
            return fill_discrete_color_scale(
                ScaleType.FILL_COLOR, value_kind, col, data_type, labels, value_map
            )
        elif scale_type == ScaleType.SIZE:
            if not size_range:
                raise GGException("expected size range")

            return fill_discrete_size_scale(
                value_kind, col, data_type, labels, value_map, size_range
            )
        elif scale_type == ScaleType.ALPHA:
            if not alpha_range:
                raise GGException("expected alpha range")

            return fill_discrete_alpha_scale(
                value_kind, col, data_type, labels, value_map, alpha_range
            )
        elif scale_type == ScaleType.LINEAR_DATA:
            if ax_kind is None:
                raise GGException("Linear data scales need an axis!")
            return fill_discrete_linear_trans_scale(
                ScaleType.LINEAR_DATA, col, ax_kind, value_kind, labels
            )
        elif scale_type == ScaleType.TRANSFORMED_DATA:
            if trans is None:
                raise GGException("Transform data needs a ScaleTransform procedure!")
            if inv_trans is None:
                raise GGException(
                    "Transform data needs an inverse ScaleTransform procedure!"
                )
            if ax_kind is None:
                raise GGException("Linear data scales need an axis!")

            return fill_discrete_linear_trans_scale(
                ScaleType.TRANSFORMED_DATA,
                col,
                ax_kind,
                value_kind,
                labels,
                trans,
                inv_trans,
            )
        elif scale_type == ScaleType.SHAPE:
            return fill_discrete_shape_scale(value_kind, col, labels, value_map)
        elif scale_type == ScaleType.TEXT:
            # TODO this is missing a bunch of required attributes as well
            return TextScale(
                gg_data=GGScaleData(
                    col=col,
                    ids=set(),  # type: ignore,
                    value_kind=VString(data=""),
                    has_discreteness=True,
                    discrete_kind=GGScaleDiscrete(
                        value_map=OrderedDict(),
                        label_seq=[],
                    ),
                    data_type=DataType.MAPPING,
                )
            )
    else:
        assert data_scale is not None, "Continuous scales require a data scale!"

        if scale_type == ScaleType.LINEAR_DATA:
            if ax_kind is None:
                raise GGException("Linear data scales need an axis!")
            return fill_continuous_linear_scale(col, ax_kind, value_kind, data_scale)
        elif scale_type == ScaleType.TRANSFORMED_DATA:
            if trans is None or inv_trans is None:
                raise GGException("expected trans and inv_trans")
            if ax_kind is None:
                raise GGException("expected axis kind")

            return fill_continuous_transformed_scale(
                col, ax_kind, value_kind, trans, inv_trans, data_scale
            )
        elif scale_type == ScaleType.COLOR:
            # TODO HIGH priority
            # double check if ColorScale.colors is incorrectly setup as List[int]
            # or if we need to do some conversion on the color
            # this is almost guaratneed to be a bug

            return fill_continuous_color_scale(
                ScaleType.COLOR, col, data_type, value_kind, data_scale, color_scale
            )
        elif scale_type == ScaleType.FILL_COLOR:
            # TODO HIGH priority
            # same as ScaleType.COLOR this is almost certain to be a bug
            return fill_continuous_color_scale(
                ScaleType.FILL_COLOR,
                col,
                data_type,
                value_kind,
                data_scale,
                color_scale,
            )
        elif scale_type == ScaleType.SIZE:
            return fill_continuous_size_scale(
                col, data_type, value_kind, data_scale, size_range
            )
        elif scale_type == ScaleType.ALPHA:
            return fill_continuous_alpha_scale(
                col, data_type, value_kind, data_scale, alpha_range
            )
        elif scale_type == ScaleType.SHAPE:
            raise ValueError("Shape not supported for continuous variables!")
        elif scale_type == ScaleType.TEXT:
            # TODO HIGH priority Same as the discrete version
            # Original version does not pass many required attrs
            # i made up some for now
            return TextScale(
                gg_data=GGScaleData(
                    col=col,
                    ids=set(),  # type: ignore,
                    value_kind=VString(data=""),
                    has_discreteness=True,
                    discrete_kind=GGScaleDiscrete(
                        value_map=OrderedDict(),
                        label_seq=[],
                    ),
                    data_type=DataType.MAPPING,
                )
            )


@dataclass
class _FillScaleData:
    scale: GGScale
    df: Optional[pd.DataFrame]


def fill_scale(
    df: pd.DataFrame, scales: List[GGScale], scale_type: ScaleType  # type: ignore
) -> List[GGScale]:
    """
    TODO refactor ASAP
    this is a mess
    """
    data: pd.Series = pd.Series(dtype=object)  # type: ignore
    trans_opt = None
    inv_trans_opt = None
    ax_kind_opt = None

    for scale in scales:
        data = pd.concat([data, df[str(scale.gg_data.col)]])  # type: ignore

    data_scale_opt = None
    label_seq_opt = None
    value_map_opt = None
    dc_kind_opt = None
    color_scale = None
    size_range = (0.0, 0.0)
    alpha_range = (0.0, 0.0)

    result = []
    for scale in scales:
        data_kind = scale.gg_data.data_type
        dc_kind_opt = scale.gg_data.discrete_kind.discrete_type
        if isinstance(scale, LinearDataScale):
            if scale.data is None:
                raise GGException("expected data")
            ax_kind_opt = scale.data.axis_kind
        elif isinstance(scale, TransformedDataScale):
            if scale.data is None:
                raise GGException("expected data")
            ax_kind_opt = scale.data.axis_kind
            trans_opt = scale.transform
            inv_trans_opt = scale.inverse_transform
        elif isinstance(scale, (ColorScaleKind, FillColorScale)):
            color_scale = scale.color_scale
        elif isinstance(scale, SizeScale):
            size_range = scale.size_range
        elif isinstance(scale, AlphaScale):
            alpha_range = scale.alpha_range

        (is_discrete, value_kind) = discrete_and_type(data, scale, dc_kind_opt)
        if isinstance(value_kind, VNull):
            print("WARNING: Unexpected data type!")
            continue

        if is_discrete:
            # todo refactor this
            discrete_kind = cast(GGScaleDiscrete, scale.gg_data.discrete_kind)
            if not discrete_kind.label_seq:
                if isinstance(scale, (LinearDataScale, TransformedDataScale)):
                    if scale.data is None:
                        raise GGException("expected data")
                    if scale.data.reversed:
                        label_seq_opt = sorted(data.unique())  # type: ignore
                    else:
                        label_seq_opt = sorted(data.unique(), reverse=True)  # type: ignore
                else:
                    label_seq_opt = sorted(data.unique())  # type: ignore
            else:
                label_seq_opt = discrete_kind.label_seq

            if discrete_kind.value_map:
                value_map_opt = discrete_kind.value_map
        else:
            discrete_kind = cast(GGScaleContinuous, scale.gg_data.discrete_kind)
            if discrete_kind.data_scale.low != discrete_kind.data_scale.high:
                data_scale_opt = discrete_kind.data_scale
            else:
                data_scale_opt = Scale(min=data.min(), max=data.max())  # type: ignore

        filled = fill_scale_impl(
            # TODO CRITICAL, this needs to be inferred from the dtype
            # infact, GGValue should probably be deleted overall eventually
            # keep for now to finish other things
            value_kind=VTODO(),
            is_discrete=is_discrete,
            col=scale.gg_data.col,
            scale_type=scale_type,
            data_type=data_kind,
            label_seq=label_seq_opt,
            value_map=value_map_opt,
            data_scale=data_scale_opt,
            ax_kind=ax_kind_opt,
            # TODO CRITICAL
            # the whole trans logic needs refactoring
            # for now ignore type
            trans=trans_opt,  # type: ignore
            # TODO CRITICAL
            # the whole inv_trans logic needs refactoring
            # for now ignore type
            inv_trans=inv_trans_opt,  # type: ignore
            # TODO CRITICAL this needs some re-weriring
            # one side is List[int] and the other is Tuple[float, float]
            # we ignore for now
            color_scale=color_scale,  # type: ignore
            size_range=size_range,
            alpha_range=alpha_range,
        )

        # TODO fix this, along with the whole function
        if isinstance(filled, (LinearDataScale, TransformedDataScale)) and isinstance(
            scale, (LinearDataScale, TransformedDataScale)
        ):
            if not (filled.data is None or scale.data is None):
                # TODO double check this is correct
                filled.data.secondary_axis = scale.data.secondary_axis
                filled.data.date_scale = scale.data.date_scale
                filled.gg_data.num_ticks = scale.gg_data.num_ticks
                filled.gg_data.breaks = scale.gg_data.breaks

            if isinstance(scale.gg_data.discrete_kind, GGScaleDiscrete) and is_discrete:
                scale.gg_data.discrete_kind.format_discrete_label = (
                    scale.gg_data.discrete_kind.format_discrete_label
                )
            elif (
                isinstance(scale.gg_data.discrete_kind, GGScaleContinuous)
                and not is_discrete
            ):
                # we can ignore type for now this whole thing will be refactored
                filled.gg_data.discrete_kind.format_continuous_label = scale.gg_data.discrete_kind.format_continuous_label  # type: ignore

        filled.gg_data.ids = scale.gg_data.ids
        result.append(filled)  # type: ignore

    return result  # type: ignore


def call_fill_scale(
    p_data: pd.DataFrame,  # type: ignore
    scales: List[_FillScaleData],
    scale_type: ScaleType,
) -> List[GGScale]:
    separate_idxs = [i for i in range(len(scales)) if scales[i].df is not None]
    scales_to_use: List[GGScale] = []

    for i, scale_data_ in enumerate(scales):
        if i not in separate_idxs:
            scales_to_use.append(scale_data_.scale)

    result: List[GGScale] = []
    if len(scales_to_use) > 0:
        filled: List[GGScale] = []
        if scales_to_use[0].scale_type == ScaleType.TRANSFORMED_DATA:
            filled = fill_scale(p_data, scales_to_use, ScaleType.TRANSFORMED_DATA)
        else:
            filled = fill_scale(p_data, scales_to_use, scale_type)

        result.extend(filled)

    # now separates
    for i in separate_idxs:
        additional: List[GGScale] = []
        if scales[i].scale.scale_type == ScaleType.TRANSFORMED_DATA:
            additional = fill_scale(
                scales[i].df,  # type: ignore
                [scales[i].scale],
                ScaleType.TRANSFORMED_DATA,
            )
        else:
            additional = fill_scale(
                scales[i].df, [scales[i].scale], scale_type  # type: ignore
            )

        if len(additional) > 1:
            raise GGException("expected at most 1 additional")
        result.extend(additional)

    return result


def add_facets(filled_scales: FilledScales, plot: GgPlot):
    if plot.facet is None:
        raise GGException("expected a facet")
    facet = plot.facet

    for fc in facet.columns:

        # TODO CRITICAL
        # this needs re-visiting very soon
        scale_ = AbstractGGScale(
            gg_data=GGScaleData(
                ids=set(range(65536)),
                col=fc,
                name=fc,
                has_discreteness=True,
                discrete_kind=GGScaleDiscrete(value_map={}, label_seq=[]),  # type: ignore
            )
        )
        fill_scale_data = _FillScaleData(scale=scale_, df=None)

        result = call_fill_scale(plot.data, [fill_scale_data], ScaleType.LINEAR_DATA)
        filled_scales.facets.extend(result)


def collect(plot: GgPlot, field_name: str) -> List[_FillScaleData]:
    scale_data: List[_FillScaleData] = []

    attr_value = getattr(plot.aes, field_name, None)
    if attr_value is not None:
        element = _FillScaleData(df=None, scale=attr_value)
        scale_data.append(element)

    # Check all geoms
    for geom in plot.geoms:
        geom_aes = getattr(geom.gg_data.aes, field_name, None)
        if geom_aes is None:
            continue

        element = _FillScaleData(df=geom.gg_data.data, scale=geom_aes)
        scale_data.append(element)

    return scale_data


def collect_scales(plot: GgPlot) -> FilledScales:
    result: Dict[Any, Any] = {}

    def fill_field(field_name: str, arg: List[GGScale]) -> None:
        if len(arg) > 0 and arg[0].gg_data.ids == set(range(0, 65535)):  # type: ignore
            result[field_name] = MainAddScales(main=arg[0], more=arg[1:])
        else:
            result[field_name] = MainAddScales(main=None, more=arg)

    xs = collect(plot, "x")
    x_filled = call_fill_scale(plot.data, xs, ScaleType.LINEAR_DATA)
    fill_field("x", x_filled)

    if any(x.scale.is_reversed() for x in xs):
        result["reversed_x"] = True
    if any(x.is_discrete() for x in x_filled):
        result["discrete_x"] = True

    xs_min = collect(plot, "x_min")
    x_min_filled = call_fill_scale(plot.data, xs_min, ScaleType.LINEAR_DATA)
    fill_field("x_min", x_min_filled)

    xs_max = collect(plot, "x_max")
    x_max_filled = call_fill_scale(plot.data, xs_max, ScaleType.LINEAR_DATA)
    fill_field("x_max", x_max_filled)

    ys = collect(plot, "y")
    y_filled = call_fill_scale(plot.data, ys, ScaleType.LINEAR_DATA)
    fill_field("y", y_filled)

    if any(y.scale.is_reversed() for y in ys):
        result["reversed_y"] = True
    if any(y.is_discrete() for y in y_filled):
        result["discrete_y"] = True

    ys_min = collect(plot, "y_min")
    y_min_filled = call_fill_scale(plot.data, ys_min, ScaleType.LINEAR_DATA)
    fill_field("y_min", y_min_filled)

    # Handle y_max scales
    ys_max = collect(plot, "y_max")
    y_max_filled = call_fill_scale(plot.data, ys_max, ScaleType.LINEAR_DATA)
    fill_field("y_max", y_max_filled)

    # Handle y_ridges scales
    ys_ridges = collect(plot, "y_ridges")
    y_ridges_filled = call_fill_scale(plot.data, ys_ridges, ScaleType.LINEAR_DATA)
    fill_field("y_ridges", y_ridges_filled)

    # Handle color scales
    colors = collect(plot, "color")
    color_filled = call_fill_scale(plot.data, colors, ScaleType.COLOR)
    fill_field("color", color_filled)

    # Handle fill scales
    fills = collect(plot, "fill")
    fill_filled = call_fill_scale(plot.data, fills, ScaleType.FILL_COLOR)
    fill_field("fill", fill_filled)

    # Handle alpha scales
    alphas = collect(plot, "alpha")
    alpha_filled = call_fill_scale(plot.data, alphas, ScaleType.ALPHA)
    fill_field("alpha", alpha_filled)

    # Handle size scales
    sizes = collect(plot, "size")
    size_filled = call_fill_scale(plot.data, sizes, ScaleType.SIZE)
    fill_field("size", size_filled)

    # Handle shape scales
    shapes = collect(plot, "shape")
    shape_filled = call_fill_scale(plot.data, shapes, ScaleType.SHAPE)
    fill_field("shape", shape_filled)

    # Handle width scales
    widths = collect(plot, "width")
    width_filled = call_fill_scale(plot.data, widths, ScaleType.LINEAR_DATA)
    fill_field("width", width_filled)

    # Handle height scales
    heights = collect(plot, "height")
    height_filled = call_fill_scale(plot.data, heights, ScaleType.LINEAR_DATA)
    fill_field("height", height_filled)

    # Handle text scales (dummy scale, only care about column)
    texts = collect(plot, "text")
    text_filled = call_fill_scale(plot.data, texts, ScaleType.TEXT)
    fill_field("text", text_filled)

    # Handle weight scales
    weights = collect(plot, "weight")
    weight_filled = call_fill_scale(plot.data, weights, ScaleType.LINEAR_DATA)
    fill_field("weight", weight_filled)

    filled_scales_result = FilledScales(**result)  # type: ignore
    if plot.facet is not None:
        add_facets(filled_scales_result, plot)

    post_process_scales(filled_scales_result, plot)
    return filled_scales_result
