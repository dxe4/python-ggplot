from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, OrderedDict, TypeVar, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from python_ggplot.core.objects import GGException


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


class ColumnType(Enum):
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


def pandas_series_to_column(series: pd.Series[Any]) -> ColumnType:
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
class BaseColumn(ABC):
    # len: int = 0 TODO

    @abstractmethod
    def col_type(self) -> ColumnType:
        pass


@dataclass
class FloatColumn(BaseColumn):
    data: NDArray[np.float64]

    def col_type(self) -> ColumnType:
        return ColumnType.FLOAT


@dataclass
class IntColumn(BaseColumn):
    data: NDArray[np.int64]

    def col_type(self) -> ColumnType:
        return ColumnType.INT


@dataclass
class BoolColumn(BaseColumn):
    data: NDArray[np.bool_]

    def col_type(self) -> ColumnType:
        return ColumnType.BOOL


@dataclass
class StringColumn(BaseColumn):
    data: NDArray[np.str_]

    def col_type(self) -> ColumnType:
        return ColumnType.STRING


@dataclass
class ObjectColumn(BaseColumn):
    data: NDArray[Any]

    def col_type(self) -> ColumnType:
        return ColumnType.OBJECT


@dataclass
class ConstantColumn(BaseColumn):
    data: Any

    def col_type(self) -> ColumnType:
        return ColumnType.CONSTANT


@dataclass
class GenericColumn(BaseColumn):
    """
    TODO
    """

    def col_type(self) -> ColumnType:
        return ColumnType.GENERIC


Column = Union[
    FloatColumn,
    IntColumn,
    BoolColumn,
    StringColumn,
    ObjectColumn,
    ConstantColumn,
    GenericColumn,
]


ColumnLike = TypeVar("ColumnLike")


class FormulaType(Enum):
    VARIABLE = auto()
    ASSIGN = auto()
    VECTOR = auto()
    SCALAR = auto()
    NONE = auto()


@dataclass
class Formula(ABC):
    name: str
    kind: FormulaType

    @abstractmethod
    def formula_type(self) -> FormulaType:
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Any:
        pass


@dataclass
class VariableFormula(Formula):
    val: GGValue

    def formula_type(self) -> FormulaType:
        return FormulaType.VARIABLE

    def evaluate(self, df: pd.DataFrame) -> Any:
        return None


@dataclass
class AssignFormula(Formula):
    lhs: str
    rhs: GGValue

    def formula_type(self) -> FormulaType:
        return FormulaType.ASSIGN

    def evaluate(self, df: pd.DataFrame) -> Any:
        return df[str(self.rhs)]  # type: ignore


@dataclass
class VectorFormula(Formula):
    col_name: str
    res_type: Any
    fn_v: Callable[[pd.DataFrame], Any]

    def formula_type(self) -> FormulaType:
        return FormulaType.VECTOR

    def evaluate(self, df: pd.DataFrame) -> Any:
        return self.fn_v(df)


@dataclass
class ScalarFormula(Formula):
    val_name: str
    val_kind: Any
    fn_s: Callable[[pd.DataFrame], GGValue]

    def reduce_(self, df: pd.DataFrame) -> GGValue:
        return self.fn_s(df)

    def formula_type(self) -> FormulaType:
        return FormulaType.SCALAR

    def evaluate(self, df: pd.DataFrame) -> Any:
        # call constantColumn(self.fn_v(df))
        return self.fn_s(df)


@dataclass
class NoneFormula(Formula):

    def formula_type(self) -> FormulaType:
        return FormulaType.NONE

    def evaluate(self, df: pd.DataFrame) -> Any:
        # newColumn(colNone, df.len)
        return None


class FormulaNode:
    kind: Formula
    name: str = ""

    def is_column(self):
        """
        # TODO high priority / urgent
        the nim logic is:
        if node.isColumn(df):
            result = df[node.val.toStr]
        else:
            result = C.constantColumn(node.val, df.len)
        we need to figure out
        a) how datamancer does consts
        b) how pandas does them internally
        for now we support column only, need to get it working first
        """
        return True

    def evaluate(self, df: pd.DataFrame) -> Any:
        # https://github.com/SciNim/Datamancer/blob/47ba4d81bf240a7755b73bc48c1cec9b638d18ae/src/datamancer/dataframe.nim#L2529
        return self.kind.evaluate(df)

    def __call__(self) -> float:
        # TODO this is used for transformation
        raise GGException()


def series_is_int(series: pd.Series[Any]) -> bool:
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


def series_is_float(series: pd.Series[Any]) -> bool:
    return str(series.dtype) in [
        "float16",
        "float32",
        "float64",
    ]


def series_is_bool(series: pd.Series[Any]):
    return str(series.dtype) == "bool"


def series_is_str(series: pd.Series[Any]):
    return str(series.dtype) == "string"


def series_is_obj(series: pd.Series[Any]):
    return str(series.dtype) == "object"


def series_value_type(series: pd.Series[Any]):
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
