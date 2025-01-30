from copy import deepcopy
from dataclasses import field
from typing import List, Optional, Set, Literal
import pandas as pd

from python_ggplot.core.units.objects import Quantity
from python_ggplot.gg.geom import Geom, GeomData, GeomPoint
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
    Theme
)
from python_ggplot.gg.utils import assign_identity_scales_get_style

STAT_TYPE_VALUES = Literal["identity", "count", "bin", "smooth"]
BIN_POSITION_VALUES = Literal["none", "center", "left", "right"]
POSITION_VALUES = Literal["identity", "stack", "dodge", "fill"]
BIN_BY_VALUES = Literal["full", "subset"]

id_counter = 1

def increment_id():
    global id_counter
    id_counter += 1

def get_gid():
    global id_counter
    id_counter += 1
    return id_counter - 1


def aes(
    x: Optional[str]=None,
    y: Optional[str]=None,
    color: Optional[str]=None,
    fill: Optional[str]=None,
    shape: Optional[str]=None,
    size: Optional[str]=None,
    xmin: Optional[str]=None,
    xmax: Optional[str]=None,
    ymin: Optional[str]=None,
    ymax: Optional[str]=None,
    width: Optional[str]=None,
    height: Optional[str]=None,
    text: Optional[str]=None,
    yridges: Optional[str]=None,
    weight: Optional[str]=None,
):
    pass

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
        theme=Theme(
            discrete_scale_margin=Quantity.centimeters(0.2)
        ),
    )
    return result

def geom_point(
    aes: Aesthetics=field(default_factory=Aesthetics),
    data: pd.DataFrame = field(default_factory=pd.DataFrame),
    color: PossibleColor=None,
    size: PossibleFloat=None,
    marker: PossibleMarker=None,
    stat: STAT_TYPE_VALUES="identity",
    bins: int=-1,
    bin_width: float=0.0,
    breaks: List[float]=field(default_factory=list),
    bin_position: BIN_POSITION_VALUES = "none",
    position: POSITION_VALUES = "identity",
    bin_by: BIN_BY_VALUES = "full",
    density: bool=False,
    alpha: Optional[float]=None
) -> 'Geom':
    '''
    TODO CRITICAL
    this is an easy fix, but it needs some thinking what is the best design
    many objects are supposed to have "required attributes"
    but they get initialised empty
    for now this is fine, we could make those params optional
    an example of this is:
        stat_kind=StatKind.create_from_enum(stat_)
    this will blow up on StatBin creation because num_bins is required
    this is fine for now, but need to provide a generic solution
    '''
    df_opt = data if len(data) > 0 else None
    bin_position_ = BinPositionType.eitem(bin_position)
    stat_ = StatType.eitem(stat)
    bin_position_ = BinPositionType.eitem(bin_position)
    position_ = PositionType.eitem(position)
    bin_by_ = BinByType.eitem(bin_by)

    # modify `Aesthetics` for all identity scales (column references) & generate style
    style = assign_identity_scales_get_style(
        aes=aes,
        p_color=color,
        p_size=size,
        p_marker=marker,
        p_alpha=alpha
    )

    gid = get_gid()
    gg_data = GeomData(
        gid=gid,
        data=df_opt,
        user_style=style,
        aes=fill_ids(aes, {gid}),
        bin_position=bin_position_,
        stat_kind=StatKind.create_from_enum(stat_),  # TODO see func docstring
        position=position_
    )
    result = GeomPoint(gg_data=gg_data)

    Geom.assign_bin_fields(
        result,
        stat_,
        bins,
        bin_width,
        breaks,
        bin_by_,
        density
    )
    return result
