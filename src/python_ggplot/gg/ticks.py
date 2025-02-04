import math
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from python_ggplot.core.common import linspace
from python_ggplot.core.coord.objects import (
    Coord1D,
    RelativeCoordType,
    StrHeightCoordType,
    TextCoordData,
)
from python_ggplot.core.objects import AxisKind, Font, GGException, Scale, TextAlignKind
from python_ggplot.gg.datamancer_pandas_compat import VectorCol
from python_ggplot.gg.scales.base import (
    FilledScales,
    GGScale,
    GGScaleContinuous,
    GGScaleDiscrete,
    LinearDataScale,
    ScaleTransformFunc,
    ScaleType,
    TransformedDataScale,
)
from python_ggplot.gg.types import DateTickAlgorithmType, DiscreteType, GgPlot, Theme
from python_ggplot.graphics.initialize import (
    TickLabelsInput,
    calc_tick_locations,
    init_ticks,
    tick_labels,
    tick_labels_from_coord,
)
from python_ggplot.graphics.objects import GraphicsObject, format_tick_value
from python_ggplot.graphics.views import ViewPort


def get_ticks(scale: Union[GGScale, Scale]) -> int:
    """
    GGscale will have num_ticks
    Scale doesn't have the attr so in that case it defaults to 10
    TODO triple check the logic down the line, sounds wrong
    """
    return getattr(scale, "num_ticks", 10)


def get_x_ticks(scale: FilledScales) -> int:
    return get_ticks(scale.get_x_scale())


def get_y_ticks(scale: FilledScales) -> int:
    return get_ticks(scale.get_y_scale())


def smallest_pow(inv_trans: ScaleTransformFunc, x: float):
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


def largest_pow(inv_trans: ScaleTransformFunc, x: float):
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
    scale_type: ScaleType,
    num_ticks: int,
    theme: Theme,
    breaks: Optional[List[float]] = None,
    trans: Optional[Callable[[float], float]] = None,
    inv_trans: Optional[Callable[[float], float]] = None,
    sec_axis_trans: Optional[VectorCol] = None,
    format_func: Optional[Callable[[float], str]] = None,
    is_secondary: bool = False,
    hide_tick_labels: bool = False,
    margin: Optional[Coord1D] = None,
) -> List[GraphicsObject]:
    """todo refactor / clean / reuse medium priority"""
    breaks = breaks or []

    bound_scale = _bound_scale(data_scale, theme, ax_kind, is_secondary)
    sec_axis_trans = sec_axis_trans or VectorCol(col_name="")

    if ax_kind == AxisKind.X:
        # todo refacotr this
        # high priority
        rotate = theme.x_ticks_rotate
        align_to = theme.x_ticks_text_align
    else:
        rotate = theme.y_ticks_rotate
        align_to = theme.y_ticks_text_align

    if scale_type == ScaleType.LINEAR_DATA:
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

    elif scale_type == ScaleType.TRANSFORMED_DATA:
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


def handle_discrete_ticks(
    view: ViewPort,
    plot: GgPlot,
    ax_kind: AxisKind,
    label_seq: Sequence[Any],  # typle Value TODO
    theme: Theme,
    is_secondary: bool = False,
    hide_tick_labels: bool = False,
    center_ticks: bool = True,
    margin: Optional[Coord1D] = None,
    format_func: Callable[[Any], str] = str,
) -> List[GraphicsObject]:
    """todo refactor / clean / reuse medium priority"""

    if is_secondary:
        raise Exception("Secondary axis for discrete axis not yet implemented!")

    num_ticks = len(label_seq)
    tick_labels: List[str] = []
    tick_locs: List[Coord1D] = []

    discr_margin = 0.0
    if theme.discrete_scale_margin is not None:
        if ax_kind == AxisKind.X:
            discr_margin = theme.discrete_scale_margin.to_relative(
                length=view.point_width()
            ).val
        elif ax_kind == AxisKind.Y:
            discr_margin = theme.discrete_scale_margin.to_relative(
                length=view.point_height()
            ).val
        else:
            raise GGException("Axis has to be x or y")

    bar_view_width = (1.0 - 2 * discr_margin) / num_ticks
    center_pos = bar_view_width / 2.0

    if not center_ticks:
        if ax_kind == AxisKind.X:
            center_pos = 0.0
        elif ax_kind == AxisKind.Y:
            center_pos = bar_view_width
        else:
            raise GGException("Axis has to be x or y")

    for i in range(num_ticks):
        if not hide_tick_labels:
            tick_labels.append(format_func(label_seq[i]))
        else:
            tick_labels.append("")

        pos = discr_margin + i * bar_view_width + center_pos
        tick_locs.append(RelativeCoordType(pos=pos))

    rotate = None
    align_to = None
    if ax_kind == AxisKind.X:
        rotate = theme.x_ticks_rotate
        align_to = theme.x_ticks_text_align
    elif ax_kind == AxisKind.Y:
        rotate = theme.y_ticks_rotate
        align_to = theme.y_ticks_text_align
    else:
        raise GGException("Axis has to be x or y")

    tick_objs, lab_objs = tick_labels_from_coord(
        view,
        tick_locs,
        tick_labels,
        ax_kind,
        rotate=rotate,
        align_override=align_to,
        font=theme.tick_label_font,
        margin=margin,
    )

    if not hide_tick_labels:
        for i in tick_objs + lab_objs:
            view.add_obj(i)

    return list(tick_objs)


def get_tick_label_margin(view: ViewPort, theme: Theme, ax_kind: AxisKind):
    """todo refactor / clean / reuse medium priority"""
    margin = 0.0
    if ax_kind == AxisKind.X:
        margin = theme.x_tick_label_margin or 1.75
    elif ax_kind == AxisKind.Y:
        margin = theme.y_tick_label_margin or -1.25
    else:
        raise GGException("expected axis x or y")

    # if no default font, use 8pt
    font = theme.tick_label_font or Font(size=8.0)
    result = StrHeightCoordType(
        pos=margin,
        data=TextCoordData(
            text="M",
            font=font,
        ),
    )

    if ax_kind == AxisKind.X:
        return result.to_relative(length=view.point_height())
    elif ax_kind == AxisKind.Y:
        return result.to_relative(length=view.point_width())
    else:
        raise GGException("expected axis kind x or y")


T = TypeVar("T")


def without_idxs(seq: List[T], idxs: Set[int]) -> List[T]:
    return [x for i, x in enumerate(seq) if i not in idxs]


def remove_ticks_within_spacing(
    ticks: List[float], tick_labels: List[str], date_spacing_in_seconds: int
) -> Tuple[List[float], List[str]]:
    cur_dur = datetime.fromtimestamp(int(ticks[0]), timezone.utc)
    last_dist: Optional[int] = None
    idxs_to_delete: Set[int] = set()

    for i in range(1, len(ticks)):
        tp_u = datetime.fromtimestamp(int(ticks[i]), timezone.utc)
        time_diff: int = int(abs((tp_u - cur_dur).total_seconds()))

        if time_diff >= date_spacing_in_seconds:
            if last_dist and last_dist < abs(time_diff - date_spacing_in_seconds):
                idxs_to_delete.discard(i - 1)
                idxs_to_delete.add(i)
                cur_dur = datetime.fromtimestamp(int(ticks[i - 1]), timezone.utc)
            else:
                cur_dur = tp_u  # new start point this index
        else:
            idxs_to_delete.add(i)  # else delete index

        last_dist = abs(time_diff - date_spacing_in_seconds)

    filtered_ticks = without_idxs(ticks, idxs_to_delete)
    filtered_tick_labels = without_idxs(tick_labels, idxs_to_delete)

    return filtered_ticks, filtered_tick_labels


def compute_tick_pos_by_date_spacing(
    first_tick: datetime, last_tick: datetime, date_spacing: timedelta
) -> List[datetime]:
    result = [first_tick]
    t = first_tick
    while t < last_tick:
        t = t + date_spacing
        result.append(t)
    return result


def handle_date_scale_ticks(
    view: ViewPort,
    plot: GgPlot,
    ax_kind: AxisKind,
    scale: Union[LinearDataScale, TransformedDataScale],
    theme: Theme,
    hide_tick_labels: bool = False,
    margin: Optional[Coord1D] = None,
) -> List[GraphicsObject]:
    """
    TODO, we pass scale_data: LinearAndTransformScaleData until we fix the objects relations
    high priority, after functionality is ported refactor#
    +overall refactor
    """

    if ax_kind == AxisKind.X:
        rotate: float = theme.x_ticks_rotate
        align_to: TextAlignKind = theme.x_ticks_text_align
    elif ax_kind == AxisKind.Y:
        rotate: float = theme.y_ticks_rotate
        align_to: TextAlignKind = theme.y_ticks_text_align
    else:
        raise GGException("expected x / y axis")

    if scale.data is None:
        raise GGException("expected scale data")
    date_scale = scale.data.date_scale
    if date_scale is None:
        # todo double check this logic (if its optional)
        raise GGException("Expected a date scale")

    tick_labels = []
    tick_pos_unix = []

    if date_scale.date_algo == DateTickAlgorithmType.FILTER:
        # TODO eventually move this logic on the scale class
        data = []
        if date_scale.is_timestamp:
            data = [
                datetime.fromtimestamp(x, timezone.utc)  # type: ignore
                for x in plot.data[scale.get_col_name()]  # type: ignore
            ]
        else:
            data = [
                date_scale.parse_date(x)  # type: ignore
                for x in plot.data[scale.get_col_name()]  # type: ignore
            ]

        tick_labels = list(
            dict.fromkeys(d.strftime(date_scale.format_string) for d in data)
        )
        tick_pos_unix = [
            datetime.strptime(label, date_scale.format_string).timestamp()
            for label in tick_labels
        ]

        remove_ticks_within_spacing(
            tick_pos_unix, tick_labels, int(date_scale.date_spacing.total_seconds())
        )

    elif date_scale.date_algo == DateTickAlgorithmType.ADD_DURATION:
        # TODO eventually move this logic on the scale class
        if date_scale.is_timestamp:
            timestamps = plot.data[scale.get_col_name()]  # type: ignore
            first_tick = datetime.fromtimestamp(timestamps.min(), timezone.utc)  # type: ignore
            last_tick = datetime.fromtimestamp(timestamps.max(), timezone.utc)  # type: ignore
        else:
            dates = sorted([date_scale.parse_date(x) for x in plot.data[scale.get_col_name()]])  # type: ignore TODO
            first_tick = min(dates)
            last_tick = max(dates)

        tick_pos = compute_tick_pos_by_date_spacing(
            first_tick, last_tick, date_scale.date_spacing
        )

        tick_labels = list(
            dict.fromkeys(d.strftime(date_scale.format_string) for d in tick_pos)
        )
        # Convert back to unix timestamps
        tick_pos_unix = [
            datetime.strptime(label, date_scale.format_string).timestamp()
            for label in tick_labels
        ]

    elif date_scale.date_algo == DateTickAlgorithmType.CUSTOM_BREAKS:
        # TODO eventually move this logic on the scale class
        tick_pos_unix = date_scale.breaks
        if not tick_pos_unix:
            raise ValueError(
                "date_algo is CUSTOM_BREAKS, but no breaks are given in the call to scale_x/y_date."
            )

        tick_labels = list(
            dict.fromkeys(
                datetime.fromtimestamp(x, timezone.utc).strftime(
                    date_scale.format_string
                )
                for x in tick_pos_unix
            )
        )

    # TODO fix this cast
    tick_coord = to_coord_1d(
        tick_pos_unix,
        ax_kind,
        cast(GGScaleContinuous, scale.gg_data.discrete_kind).data_scale,
    )

    tick_objs, lab_objs = tick_labels_from_coord(
        view,
        tick_coord,
        tick_labels,
        ax_kind,
        is_secondary=False,
        rotate=rotate,
        align_override=align_to,
        font=theme.tick_label_font,
        margin=margin or Coord1D.create_relative(0.0),
    )

    if not hide_tick_labels:
        for i in tick_objs + lab_objs:
            view.add_obj(i)

    return list(tick_objs)


def handle_ticks(
    view: ViewPort,
    filled_scales: FilledScales,
    plot: GgPlot,
    ax_kind: AxisKind,
    theme: Theme,
    hide_tick_labels: bool = False,
    num_ticks_opt: Optional[int] = None,
    bound_scale_opt: Optional[Scale] = None,
):
    """TODO clean this up later"""

    margin_opt: Optional[Coord1D] = None
    scale: Optional[GGScale] = None
    num_ticks: int = 10

    if ax_kind == AxisKind.X:
        scale = filled_scales.get_scale(filled_scales.x)
        num_ticks = num_ticks_opt if num_ticks_opt is not None else get_ticks(scale)
        if theme.x_tick_label_margin is not None:
            margin_opt = get_tick_label_margin(view, theme, ax_kind)
    elif ax_kind == AxisKind.Y:
        scale = filled_scales.get_scale(filled_scales.x)
        num_ticks = num_ticks_opt if num_ticks_opt is not None else get_ticks(scale)
        if theme.y_tick_label_margin is not None:
            margin_opt = get_tick_label_margin(view, theme, ax_kind)
    else:
        raise GGException("expect x/y axis")

    if not isinstance(scale, (LinearDataScale, TransformedDataScale)):
        raise GGException(
            f"Expected LinearData or TransformedData received {scale.__class__.__name__}"
        )

    has_scale = len(scale.gg_data.col.col_name) > 0 if scale else False
    result: List[GraphicsObject] = []

    scale_discrete_kind = scale.gg_data.discrete_kind

    if has_scale:
        if isinstance(scale_discrete_kind, GGScaleDiscrete):
            format_fn = scale_discrete_kind.format_discrete_label or (lambda x: str(x))  # type: ignore

            if scale.data is not None and scale.data.date_scale is not None:
                result = handle_date_scale_ticks(
                    view,
                    plot,
                    ax_kind,
                    scale,
                    theme,
                    hide_tick_labels,
                    margin_opt,
                )

            else:
                result = handle_discrete_ticks(
                    view,
                    plot,
                    ax_kind,
                    scale.gg_data.discrete_kind.label_seq,  # type: ignore  TODO fix
                    theme=theme,
                    hide_tick_labels=hide_tick_labels,
                    margin=margin_opt,
                    format_func=format_fn,  # type: ignore
                )

            if filled_scales.has_secondary(ax_kind):
                additional_ticks = handle_discrete_ticks(
                    view,
                    plot,
                    ax_kind,
                    scale_discrete_kind.label_seq,  # type: ignore
                    theme=theme,
                    is_secondary=True,
                    hide_tick_labels=hide_tick_labels,
                    margin=margin_opt,
                    format_func=format_fn,  # type: ignore
                )
                result.extend(additional_ticks)  # type: ignore

        elif isinstance(scale_discrete_kind, GGScaleContinuous):
            data_scale = get_correct_data_scale(view, ax_kind)
            if scale.data is not None and scale.data.date_scale is not None:
                result = handle_continuous_ticks(
                    view,
                    plot,
                    ax_kind,
                    data_scale,
                    scale.scale_type,
                    num_ticks,
                    breaks=scale.data.date_scale.breaks,  # might be empty
                    trans=scale.transform,
                    inv_trans=scale.inverse_transform,  # type: ignore TODO this needs some more work
                    theme=theme,
                    hide_tick_labels=hide_tick_labels,
                    margin=margin_opt,
                    format=scale.format_continuous_label,  # type: ignore TODO this needs more work
                )
            else:
                result = handle_date_scale_ticks(
                    view, plot, ax_kind, scale, theme, hide_tick_labels, margin_opt
                )

            if filled_scales.has_secondary(ax_kind):
                sec_axis = filled_scales.get_secondary_axis(ax_kind)
                # TODO fix this too
                trans = (
                    sec_axis.transform  # type: ignore
                    if sec_axis.scale.scale_type == ScaleType.LINEAR_DATA
                    else None
                )
                trans_fn = (
                    sec_axis.transform  # type: ignore
                    if sec_axis.scale.scale_type == ScaleType.TRANSFORMED_DATA
                    else None
                )
                inv_trans_fn = (
                    sec_axis.inverse_transform  # type: ignore
                    if sec_axis.scale.scale_type == ScaleType.TRANSFORMED_DATA
                    else None
                )

                if scale.data is None or scale.data.date_scale is None:
                    # TODO double check this
                    raise GGException("expected data and date scale")

                handle_continuous_ticks(
                    view,
                    plot,
                    ax_kind,
                    data_scale,
                    sec_axis.scale.scale_type,
                    num_ticks,
                    breaks=scale.data.date_scale.breaks,  # might be empty
                    trans=trans_fn,
                    inv_trans=inv_trans_fn,
                    sec_axis_trans=trans,
                    theme=theme,
                    hide_tick_labels=hide_tick_labels,
                    is_secondary=True,
                    margin=margin_opt,
                    format=scale.format_continuous_label,  # type: ignore
                )
    else:
        # if ax_kind == AxisKind.X:
        #     raise GGException("expected Y axis")
        bound_scale = bound_scale_opt or (
            theme.x_margin_range if ax_kind == AxisKind.X else theme.y_margin_range
        )
        ticks = init_ticks(view, ax_kind, num_ticks=num_ticks, bound_scale=bound_scale)
        ticks = init_ticks(view, ax_kind, num_ticks=num_ticks, bound_scale=bound_scale)

        tick_labs = tick_labels(
            view,
            ticks,
            TickLabelsInput(  # type: ignore use generics?
                font=theme.tick_label_font, margin=margin_opt
            ),
        )

        if not hide_tick_labels:
            for i in ticks + tick_labs:
                view.add_obj(i)
        result = ticks

    return list(result)
