from typing import Dict, Optional, Tuple

from python_ggplot.core.objects import AxisKind, GGException
from python_ggplot.gg.datamancer_pandas_compat import VectorCol, VNull
from python_ggplot.gg.scales.base import (
    GGScale,
    GGScaleData,
    LinearAndTransformScaleData,
    LinearDataScale,
    ScaleType,
    TransformedDataScale,
    scale_type_to_cls,
)
from python_ggplot.gg.types import Aesthetics

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
        "x_min": xmin,
        "x_max": xmax,
        "y_min": ymin,
        "y_max": ymax,
        "width": width,
        "height": height,
        "text": text,
        "y_ridges": yridges,
        "weight": weight,
    }
    return _init_aes(data)
