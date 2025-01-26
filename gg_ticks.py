import math
from typing import Callable, List, Optional, Sequence, Tuple

from python_ggplot.core.common import linspace
from python_ggplot.core.coord.objects import Coord1D, DataCoordType
from python_ggplot.core.objects import AxisKind, GGException, Scale
from python_ggplot.datamancer_pandas_compat import FormulaNode
from python_ggplot.gg_geom import FilledScales
from python_ggplot.gg_scales import (
    GGScale,
    ScaleKind,
    ScaleTransform,
    ScaleType,
    get_x_scale,
    get_y_scale,
)
from python_ggplot.gg_types import GgPlot, Theme
from python_ggplot.graphics.initialize import (
    TickLabelsInput,
    calc_tick_locations,
    tick_labels_from_coord,
)
from python_ggplot.graphics.objects import GraphicsObject, format_tick_value
from python_ggplot.graphics.views import ViewPort


def get_ticks(scale: GGScale) -> int:
    # TODO for backwards compat, we should set default on dataclass
    # unless i miss something obvious
    return scale.num_ticks or 10


def get_x_ticks(scale: FilledScales) -> int:
    return get_ticks(get_x_scale(scale))


def get_y_ticks(scale: FilledScales) -> int:
    return get_ticks(get_y_scale(scale))


def smallest_pow(inv_trans: ScaleTransform, x: float):
    if x < 0:
        raise GGException("expected positive X")

    result: float = 1.0
    exp = 0
    if x < 1.0:
        while result > x and not math.isclose(result, x):
            result = inv_trans(float(exp - 1))
            exp -= 1
    else:
        while result < x and not math.isclose(result, x):
            result = inv_trans(float(exp + 1))
            exp += 1
        result = inv_trans(float(exp - 1))
    return result


def largest_pow(inv_trans: ScaleTransform, x: float):
    if x < 0:
        raise GGException("expected positive X")

    result = 1.0
    exp = 0
    if x < 1.0:
        while result > x and not math.isclose(result, x):
            result = inv_trans(float(exp - 1))
            exp -= 1
        result = inv_trans(float(exp + 1))
    else:
        while result < x and not math.isclose(result, x):
            result = inv_trans(float(exp + 1))
            exp += 1
    return result


def compute_label(
    tick: float, tick_scale: float = 0.0, fmt: Optional[Callable[[float], str]] = None
) -> str:
    if fmt is not None:
        return fmt(tick)
    else:
        return format_tick_value(tick, tick_scale / 5.0)


def compute_labels(
    ticks: Sequence[float],
    tick_scale: float = 0.0,
    fmt: Optional[Callable[[float], str]] = None,
) -> list[str]:
    return [compute_label(tick, tick_scale, fmt) for tick in ticks]


def tick_pos_transformed(
    scale: Scale,
    trans: Callable[[float], float],
    inv_trans: Callable[[float], float],
    minv: float,
    maxv: float,
    bound_scale: Scale,
    breaks: Sequence[float] = [],
    hide_tick_labels: bool = False,
    format: Callable[[float], str] = str,
) -> Tuple[List[str], List[float]]:

    labs: list[str] = []
    lab_pos: list[float] = []

    if not breaks:
        exp = int(math.floor(bound_scale.low))
        while float(exp) < bound_scale.high:
            cur = inv_trans(float(exp))
            num_to_add = round(inv_trans(1.0))

            minors = linspace(cur, inv_trans(float(exp + 1)) - cur, int(num_to_add) - 1)
            lab_pos.extend([trans(x) for x in minors])

            if (bound_scale.high - bound_scale.low) > 1.0 or hide_tick_labels:
                if not hide_tick_labels:
                    labs.append(compute_label(cur, fmt=format))
                else:
                    labs.append("")

                labs.extend([""] * (len(minors) - 1))
            else:
                # use all minors as labelled
                labs.extend([compute_label(x, fmt=format) for x in minors])
            exp += 1

        if not hide_tick_labels:
            labs.append(compute_label(inv_trans(float(exp)), fmt=format))
        else:
            labs.append("")
        lab_pos.append(float(exp))
    else:
        lab_pos = [trans(x) for x in breaks]
        labs = [compute_label(inv_trans(x), fmt=format) for x in lab_pos]

    filter_idx = [
        i
        for i in range(len(lab_pos))
        if bound_scale.low <= lab_pos[i] <= bound_scale.high
    ]

    labs = [labs[i] for i in filter_idx]
    lab_pos = [lab_pos[i] for i in filter_idx]

    return labs, lab_pos


def tick_pos_linear(
    scale: Scale, num_ticks: int, breaks: Optional[List[float]] = None
) -> List[float]:
    if not breaks:
        new_scale, _, new_num_ticks = calc_tick_locations(scale, num_ticks)
        return linspace(new_scale.low, new_scale.high, new_num_ticks + 1)
    return breaks


def apply_bound_scale(ticks: List[float], bound_scale: Scale) -> List[float]:
    return [t for t in ticks if bound_scale.low <= t <= bound_scale.high]


def get_correct_data_scale(view: ViewPort, ax_kind: AxisKind) -> Optional[Scale]:
    return {
        AxisKind.X: view.x_scale,
        AxisKind.Y: view.y_scale,
    }[ax_kind]


def apply_scale_trans(scale: Scale, trans: Optional[Callable[[], float]]) -> Scale:
    if trans is None:
        return scale

    result = Scale(low=scale.low * trans(), high=scale.high * trans())
    return result


def revert_scale_trans(
    ticks: List[float], trans: Optional[Callable[[], float]]
) -> List[float]:
    if trans is None:
        return ticks
    return [t / trans() for t in ticks]


def to_coord_1d(ticks: List[float], ax_kind: AxisKind, scale: Scale) -> List[Coord1D]:
    return [Coord1D.create_data(t, scale, ax_kind) for t in ticks]


def _bound_scale(
    data_scale: Scale, theme: Theme, ax_kind: AxisKind, is_secondary: bool
) -> Scale:
    if is_secondary:
        return data_scale
    if ax_kind == AxisKind.X:
        return theme.x_margin_range
    elif ax_kind == AxisKind.Y:
        return theme.y_margin_range
    else:
        raise GGException("unexpected state")


def handle_continuous_ticks(
    view: ViewPort,
    p: GgPlot,
    ax_kind: AxisKind,
    data_scale: Scale,
    sc_kind: ScaleKind,
    num_ticks: int,
    theme: Theme,
    breaks: Optional[List[float]] = None,
    trans: Optional[Callable[[float], float]] = None,
    inv_trans: Optional[Callable[[float], float]] = None,
    sec_axis_trans: Optional[FormulaNode] = None,
    format_func: Optional[Callable[[float], str]] = None,
    is_secondary: bool = False,
    hide_tick_labels: bool = False,
    margin: Optional[Coord1D] = None,
) -> List[GraphicsObject]:
    breaks = breaks or []

    bound_scale = _bound_scale(data_scale, theme, ax_kind, is_secondary)
    sec_axis_trans = sec_axis_trans or FormulaNode()

    if ax_kind == AxisKind.X:
        # todo refacotr this
        # high priority
        rotate = theme.x_ticks_rotate
        align_to = theme.x_ticks_text_align
    else:
        rotate = theme.y_ticks_rotate
        align_to = theme.y_ticks_text_align

    if sc_kind.scale_type == ScaleType.LINEAR_DATA:
        scale = apply_scale_trans(data_scale, sec_axis_trans)
        bound_scale = apply_scale_trans(bound_scale, sec_axis_trans)

        ticks: List[float] = tick_pos_linear(scale, num_ticks, breaks)
        ticks: List[float] = apply_bound_scale(ticks, bound_scale)
        labels: List[str] = compute_labels(ticks, scale.high - scale.low, format_func)

        # Revert scale transformation for ticks if necessary
        ticks: List[float] = revert_scale_trans(ticks, sec_axis_trans)
        tick_coord: List[Coord1D] = to_coord_1d(ticks, ax_kind, scale)

        tick_objs, lab_objs = tick_labels_from_coord(
            view,
            tick_coord,
            labels,
            axis_kind=ax_kind,
            is_secondary=is_secondary,
            rotate=rotate,
            align_override=align_to,
            font=theme.tick_label_font,
            margin=margin,
        )

        if not hide_tick_labels:
            for i in tick_objs + lab_objs:
                view.add_obj(i)

        return list(tick_objs)

    elif sc_kind.scale_type == ScaleType.TRANSFORMED_DATA:
        if inv_trans is None:
            raise GGException("expected inv_trans")

        if trans is None:
            raise GGException("expected trans")

        scale = apply_scale_trans(data_scale, sec_axis_trans)
        min_val = smallest_pow(inv_trans, inv_trans(data_scale.low))
        max_val = largest_pow(inv_trans, inv_trans(data_scale.high))

        labs, label_pos = tick_pos_transformed(
            scale,
            trans,
            inv_trans,
            min_val,
            max_val,
            bound_scale,
            breaks=breaks,
            hide_tick_labels=hide_tick_labels,
            format=format_func,  # type: ignore TODO FIX
        )

        tick_locs = to_coord_1d(label_pos, ax_kind, data_scale)

        # Update view scale
        if ax_kind == AxisKind.X:
            low, high = (trans(min_val), trans(max_val))
            view.x_scale = Scale(low=low, high=high)
        else:
            low, high = (trans(min_val), trans(max_val))
            view.y_scale = Scale(low=low, high=high)

        tick_objs, lab_objs = tick_labels_from_coord(
            view,
            tick_locs,
            labs,
            ax_kind,
            is_secondary=is_secondary,
            rotate=rotate,
            align_override=align_to,
            font=theme.tick_label_font,
            margin=margin,
        )

        if not hide_tick_labels:
            for i in tick_objs + lab_objs:
                view.add_obj(i)

        return list(tick_objs)

    return []
