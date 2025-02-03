from dataclasses import dataclass
from enum import auto
from typing import Any, Dict, Optional, OrderedDict

import pandas as pd

from python_ggplot.core.objects import GGEnum, GGException


class ColumnType(GGEnum):
    NONE = auto()
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    STRING = auto()
    OBJECT = auto()
    CONSTANT = auto()
    GENERIC = auto()

    def is_scalar(self):
        return self in {ColumnType.FLOAT, ColumnType.INT}


PANDAS_TYPES: Dict[str, ColumnType] = {
    "int64": ColumnType.INT,
    "int32": ColumnType.INT,
    "float64": ColumnType.FLOAT,
    "float32": ColumnType.FLOAT,
    "bool": ColumnType.BOOL,
    "object": ColumnType.OBJECT,  # todo
    "string": ColumnType.STRING,
    # TODO generic or object?
    "category": ColumnType.OBJECT,
    # TODO why no DT type in data mancer? figure this out
    "datetime64[ns]": ColumnType.OBJECT,
    "datetime64[ns, tz]": ColumnType.OBJECT,
    # TODO why no TD type in data mancer? figure this out
    "timedelta64[ns]": ColumnType.OBJECT,
    # TODO figure this out too, complex numbers maybe of lower priority
    "complex64": ColumnType.OBJECT,
    "complex128": ColumnType.OBJECT,
}


def pandas_series_to_column(series: pd.Series) -> ColumnType:
    """
    TODO this is incomplete impl
    but this will allow to port the other logic
    and as we go along we adapt it
    once we find cases of datetimes we come back here and add it
    """
    if series.isna().all():  # type: ignore
        # TODO is there a better way for this?
        # can this cause bottle neck? maybe its fine
        return ColumnType.NONE
    result = PANDAS_TYPES[str(series.dtype)]
    return result


@dataclass
class GGValue:
    """
    TODO high priority. find a proxy mechanism for this
    """


@dataclass
class VString(GGValue):
    data: str


@dataclass
class VInt(GGValue):
    data: int


@dataclass
class VFloat(GGValue):
    data: float


@dataclass
class VBool(GGValue):
    data: bool


@dataclass
class VObject(GGValue):
    fields: OrderedDict[str, GGValue]


@dataclass
class VNull(GGValue):
    pass


@dataclass
class VTODO(GGValue):
    pass


@dataclass
class VLinearData(GGValue):
    """
    TODO this will need some changing down the line
    most likely the whole GGValue concept
    """

    data: GGValue


@dataclass
class VFillColor(GGValue):
    """
    TODO this will need some changing down the line
    most likely the whole GGValue concept
    """

    data: GGValue


@dataclass
class VectorCol:
    col_name: str
    res_type: Optional[Any] = None

    def __str__(self) -> str:
        return self.col_name

    def evaluate(self, df: pd.DataFrame) -> Any:
        return df[self.col_name]  # type: ignore

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # TODO handle_continuous_ticks
        raise GGException(
            "WARNING you called vector col, did you intend for formula node?"
        )


def series_is_int(series: pd.Series) -> bool:
    return str(series.dtype) in [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]


def series_is_float(series: pd.Series) -> bool:
    return str(series.dtype) in [
        "float16",
        "float32",
        "float64",
    ]


def series_is_bool(series: pd.Series) -> bool:
    return str(series.dtype) == "bool"


def series_is_str(series: pd.Series) -> bool:
    return str(series.dtype) == "string"


def series_is_obj(series: pd.Series) -> bool:
    return str(series.dtype) == "object"


def series_value_type(series: pd.Series) -> str:
    dtype = str(series.dtype)
    if dtype in [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]:
        return "int"
    elif dtype in [
        "float16",
        "float32",
        "float64",
    ]:
        return "float"
    elif dtype in ["bool"]:
        return "bool"
    elif dtype in ["string"]:
        return "string"
    elif dtype in ["object"]:
        return "object"

    raise GGException(f"dtype not supported {dtype}")
