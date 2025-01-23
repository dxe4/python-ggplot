from typing import cast

import numpy as np
import pandas as pd

from python_ggplot.datamancer_pandas_compat import (
    FormulaType,
    ScalarFormula,
    pandas_series_to_column,
)
from python_ggplot.gg_scales import GGScale


def add_identity_data(col: "str", df: pd.DataFrame, scale: GGScale):
    # TODO partially ported, needs some thinking
    data = pandas_series_to_column(df[col])

    if scale.col.kind.formula_type == FormulaType.SCALAR:
        temp = cast(ScalarFormula, scale.col.kind)
        contanst_col = temp.reduce_(df[col]).data
        # a scalar may happen if the user uses a reducing operation as a formula
        # for an aes, e.g. ``x = f{float -> float: getMean(`bins`, `counts`)``

        # col.add(res)  => nim version does this
        # if we add in one col, does that mean every other col gets a new row?
        # this is a case for constant,
        # so my guess is theres some special logic that replaces everything with 1 val
        res = temp.reduce_(df)
    else:
        # col.add(res)  => nim version does this
        # need to figure out exactly what is the outcome of this before adding it

        res = scale.col.evaluate(df)


def draw_sample_idx(s_high: int, num: int = 100, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    idx_num = min(num - 1, s_high)
    return np.random.randint(0, s_high + 1, size=idx_num + 1)
