from copy import deepcopy
from dataclasses import field
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import pandas as pd
from typing_extensions import Dict

from python_ggplot.common.enum_literals import (
    BIN_BY_VALUES,
    BIN_POSITION_VALUES,
    POSITION_VALUES,
    STAT_TYPE_VALUES,
)
from python_ggplot.core.objects import GGException
from python_ggplot.core.units.objects import Quantity
from python_ggplot.gg.datamancer_pandas_compat import VectorCol, VNull
from python_ggplot.gg.geom import Geom, GeomData, GeomPoint
from python_ggplot.gg.scales.base import (
    GGScale,
    GGScaleData,
    LinearAndTransformScaleData,
    LinearDataScale,
    ScaleType,
    TransformedDataScale,
    scale_type_to_cls,
)
from python_ggplot.gg.types import (
    Aesthetics,
    BinByType,
    BinPositionType,
    GgPlot,
    PositionType,
    PossibleColor,
    PossibleFloat,
    PossibleMarker,
    StatKind,
    StatType,
    Theme,
)
from python_ggplot.gg.utils import assign_identity_scales_get_style
from tests.test_view import AxisKind

id_counter = 1


def increment_id():
    global id_counter
    id_counter += 1


def get_gid():
    global id_counter
    id_counter += 1
    return id_counter - 1


CallableNoType = Callable[..., Any]


_AES_PARAM_TO_SCALE_ARGS: Dict[str, Tuple[ScaleType, Optional[AxisKind]]] = {
    "x": (ScaleType.LINEAR_DATA, AxisKind.X),
    "y": (ScaleType.LINEAR_DATA, AxisKind.Y),
    "color": (ScaleType.COLOR, None),
    "fill": (ScaleType.COLOR, None),
    "shape": (ScaleType.SHAPE, None),
    "size": (ScaleType.SIZE, None),
    "xmin": (ScaleType.LINEAR_DATA, AxisKind.X),
    "xmax": (ScaleType.LINEAR_DATA, AxisKind.X),
    "ymin": (ScaleType.LINEAR_DATA, AxisKind.Y),
    "ymax": (ScaleType.LINEAR_DATA, AxisKind.Y),
    "width": (ScaleType.LINEAR_DATA, AxisKind.X),
    "height": (ScaleType.LINEAR_DATA, AxisKind.Y),
    "text": (ScaleType.TEXT, None),
    "yridges": (ScaleType.LINEAR_DATA, AxisKind.Y),
    "weight": (ScaleType.LINEAR_DATA, AxisKind.Y),
}


def _or_none_scale(
    value: Any,
    scale_kind: ScaleType,
    axis_kind: Optional[AxisKind] = None,
    is_discrete: bool = False,
) -> Optional[GGScale]:
    if value is None:
        return None
    return GGScale(
        value=value,
        scale_kind=scale_kind,
        axis_kind=axis_kind,
        has_discreteness=is_discrete,
    )


def _init_field(arg_: str, arg_value_: Optional[str]):
    if arg_value_ is None:
        return None

    has_discreteness = has_factor(arg_)
    (scale_type, axis_kind) = _AES_PARAM_TO_SCALE_ARGS[arg_]
    scale_cls = scale_type_to_cls(scale_type)

    if scale_cls in (LinearDataScale, TransformedDataScale):
        if axis_kind is None:
            raise GGException("expected axis type")

        data = LinearAndTransformScaleData(axis_kind=axis_kind)
        return scale_cls(
            gg_data=GGScaleData(
                col=VectorCol(col_name=arg_value_),
                has_discreteness=has_discreteness,
                value_kind=VNull(),  # TODO this will be fixed eventually
            ),
            data=data,
        )
    else:
        return scale_cls(
            gg_data=GGScaleData(
                col=VectorCol(arg_value_),
                has_discreteness=has_discreteness,
                value_kind=VNull(),  # TODO this will be fixed eventually
            )
        )


def has_factor(arg_: str) -> bool:
    # TODO macro + AST magic from nim
    # skip for now, but we have to deal with this one day
    return False


def _init_aes(data: Dict[str, Optional[str]]) -> Aesthetics:

    aes_data: Dict[str, Optional[GGScale]] = {}
    for arg_, arg_value_ in data.items():
        new_scale = _init_field(arg_, arg_value_)  # type: ignore
        aes_data[arg_] = new_scale

    return Aesthetics(**aes_data)


def aes(
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    fill: Optional[str] = None,
    shape: Optional[str] = None,
    size: Optional[str] = None,
    xmin: Optional[str] = None,
    xmax: Optional[str] = None,
    ymin: Optional[str] = None,
    ymax: Optional[str] = None,
    width: Optional[str] = None,
    height: Optional[str] = None,
    text: Optional[str] = None,
    yridges: Optional[str] = None,
    weight: Optional[str] = None,
):
    """
    We dont want the public API to take in **kwargs
    that is very hard for users to read
    we'd rather take them in and call a private function underneath
    The nature of this has to be a bit more dynamic because the origin is using macro
    otherwise it will be a good amount of re-work around the logic

    TODO: provide Literal definitions for input types
    """
    data: Dict[str, Optional[str]] = {
        "x": x,
        "y": y,
        "color": color,
        "fill": fill,
        "shape": shape,
        "size": size,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": width,
        "height": height,
        "text": text,
        "yridges": yridges,
        "weight": weight,
    }
    return _init_aes(data)


def fill_ids(aes: Aesthetics, gids: Set[int]) -> Aesthetics:
    result = deepcopy(aes)

    for field, _ in result.__dataclass_fields__.items():
        value = getattr(result, field)
        if value is not None:
            value = deepcopy(value)
            value.ids = gids
            setattr(result, field, value)

    return result


def ggplot(data: pd.DataFrame, aes: Optional[Aesthetics] = None) -> GgPlot:
    shallow_copy: pd.DataFrame = data.copy(deep=False)
    if aes is None:
        aes = Aesthetics()

    aes = fill_ids(aes, set(range(0, 65536)))

    result = GgPlot(
        data=shallow_copy,
        aes=aes,
        theme=Theme(discrete_scale_margin=Quantity.centimeters(0.2)),
    )
    return result


def geom_point(
    aes: Aesthetics = field(default_factory=Aesthetics),
    data: pd.DataFrame = field(default_factory=pd.DataFrame),
    color: PossibleColor = None,
    size: PossibleFloat = None,
    marker: PossibleMarker = None,
    stat: STAT_TYPE_VALUES = "identity",
    bins: int = -1,
    bin_width: float = 0.0,
    breaks: List[float] = field(default_factory=list),
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool = False,
    alpha: Optional[float] = None,
) -> "Geom":
    """
    TODO CRITICAL
    this is an easy fix, but it needs some thinking what is the best design
    many objects are supposed to have "required attributes"
    but they get initialised empty
    for now this is fine, we could make those params optional
    an example of this is:
        stat_kind=StatKind.create_from_enum(stat_)
    this will blow up on StatBin creation because num_bins is required
    this is fine for now, but need to provide a generic solution
    """
    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)

    # modify `Aesthetics` for all identity scales (column references) & generate style
    style = assign_identity_scales_get_style(
        aes=aes, p_color=color, p_size=size, p_marker=marker, p_alpha=alpha
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),  # TODO see func docstring
        position=position_,
    )
    result = GeomPoint(gg_data=gg_data)

    Geom.assign_bin_fields(result, stat_, bins, bin_width, breaks, bin_by_, density)
    return result
