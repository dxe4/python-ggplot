"""
TODO this file will include many errors,
it will need lot of fixing once everything is in place
Smoothing in particular we may skip for alpha version
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_ggplot.common.maths import histogram
from python_ggplot.core.objects import AxisKind, GGException, Scale
from python_ggplot.gg.datamancer_pandas_compat import VectorCol, VNull, VString
from python_ggplot.gg.geom.base import (
    FilledGeom,
    FilledGeomContinuous,
    FilledGeomData,
    FilledGeomDiscrete,
    FilledGeomErrorBar,
    FilledGeomHistogram,
    Geom,
    GeomType,
    HistogramDrawingStyle,
)
from python_ggplot.gg.scales.base import (
    FilledScales,
    GGScale,
    GGScaleContinuous,
    GGScaleDiscrete,
    LinearDataScale,
    MainAddScales,
    ScaleType,
    TransformedDataScale,
)
from python_ggplot.gg.styles.utils import apply_style, change_style, use_or_default
from python_ggplot.gg.ticks import get_x_ticks, get_y_ticks
from python_ggplot.gg.types import (
    COUNT_COL,
    PREV_VALS_COL,
    SMOOTH_VALS_COL,
    BinByType,
    GgPlot,
    GGStyle,
    PositionType,
    SmoothMethodType,
    StatBin,
    StatSmooth,
    StatType,
)
from python_ggplot.graphics.initialize import calc_tick_locations
