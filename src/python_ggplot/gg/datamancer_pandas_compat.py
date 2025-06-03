from dataclasses import dataclass
from enum import auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, OrderedDict, Union

import pandas as pd

from python_ggplot.core.objects import GGEnum, GGException

if TYPE_CHECKING:
    from python_ggplot.gg.types import ColOperator, gg_col, gg_col_const, gg_col_anonymous


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


def pandas_series_to_column(series: "pd.Series[Any]") -> ColumnType:
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


class GGValue:
    """
    TODO high priority. find a proxy mechanism for this
    """


@dataclass(frozen=True)
class VString(GGValue):
    data: str


@dataclass(frozen=True)
class VInt(GGValue):
    data: int


@dataclass(frozen=True)
class VFloat(GGValue):
    data: float


@dataclass(frozen=True)
class VBool(GGValue):
    data: bool


@dataclass(frozen=True)
class VObject(GGValue):
    fields: OrderedDict[str, GGValue]


@dataclass(frozen=True)
class VNull(GGValue):
    pass


@dataclass(frozen=True)
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


def python_type_to_gg_value(value: Any) -> GGValue:
    conversion = {
        "str": VString,
        "int": VInt,
        "float": VFloat,
        "bool": VBool,
    }
    value_type = type(value).__name__
    if value_type == "NoneType":
        return VNull()

    try:
        cls = conversion[value_type]
        return cls(data=value)
    except KeyError:
        raise GGException(f"cannot convert type {value_type} to GGValue")


@dataclass
class VectorCol:
    col_name: Union[str, "gg_col", "gg_col_const", "gg_col_anonymous"]
    res_type: Optional[Any] = None
    series: Optional[pd.Series] = None  # type: ignore

    def get_transformations(self) -> Optional[List["ColOperator"]]:
        from python_ggplot.gg.types import gg_col

        if isinstance(self.col_name, gg_col):
            return self.col_name.operators
        else:
            return None

    def __str__(self) -> str:
        from python_ggplot.gg.types import gg_col, gg_col_const

        if isinstance(self.col_name, (gg_col, gg_col_const)):
            return str(self.col_name)
        else:
            return self.col_name

    def evaluate(self, df: pd.DataFrame) -> Any:
        from python_ggplot.gg.types import gg_col, gg_col_const, gg_col_anonymous

        if isinstance(self.col_name, (gg_col, gg_col_const, gg_col_anonymous)):
            return self.col_name.evaluate(df)  # type: ignore
        else:
            return df[self.col_name]  # type: ignore

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # TODO handle_continuous_ticks
        raise GGException(
            "WARNING you called vector col, did you intend for formula node?"
        )


def series_is_int(series: "pd.Series[Any]") -> bool:
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


def series_is_float(series: "pd.Series[Any]") -> bool:
    return str(series.dtype) in [
        "float16",
        "float32",
        "float64",
    ]


def series_is_bool(series: "pd.Series[Any]") -> bool:
    return str(series.dtype) == "bool"


def series_is_str(series: "pd.Series[Any]") -> bool:
    return str(series.dtype) == "string"


def series_is_obj(series: "pd.Series[Any]") -> bool:
    return str(series.dtype) == "object"  # type: ignore


def series_value_type(series: "pd.Series[Any]") -> str:
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
    elif dtype == "bool":
        return "bool"
    elif dtype == "string":
        return "string"
    elif dtype == "object":
        return "object"
    elif dtype == "category":
        return "category"

    raise GGException(f"dtype not supported {dtype}")
