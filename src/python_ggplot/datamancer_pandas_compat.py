from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, OrderedDict, TypeVar, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class GGValue:
    """
    TODO high priority. find a proxy mechanism for this
    """

    pass


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


def pandas_series_to_column(series: pd.Series) -> ColumnType:
    """
    TODO this is incomplete impl
    but this will allow to port the other logic
    and as we go along we adapt it
    once we find cases of datetimes we come back here and add it
    """
    if series.isna().all():
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


class FormulaKind(Enum):
    VARIABLE = auto()
    ASSIGN = auto()
    VECTOR = auto()
    SCALAR = auto()
    NONE = auto()


@dataclass
class Formula(Generic[ColumnLike]):
    name: str
    kind: FormulaKind


@dataclass
class VariableFormula(Formula[ColumnLike]):
    val: GGValue

    def formula_type(self) -> FormulaKind:
        return FormulaKind.VARIABLE


@dataclass
class AssignFormula(Formula[ColumnLike]):
    lhs: str
    rhs: GGValue

    def formula_type(self) -> FormulaKind:
        return FormulaKind.ASSIGN


@dataclass
class VectorFormula(Formula[ColumnLike]):
    col_name: str
    res_type: Any
    fn_v: Callable[[pd.Series], ColumnLike]

    def formula_type(self) -> FormulaKind:
        return FormulaKind.VECTOR


@dataclass
class ScalarFormula(Formula[ColumnLike]):
    val_name: str
    val_kind: Any
    fn_s: Callable[[pd.Series], GGValue]

    def formula_type(self) -> FormulaKind:
        return FormulaKind.SCALAR


@dataclass
class NoneFormula(Formula[ColumnLike]):

    def formula_type(self) -> FormulaKind:
        return FormulaKind.NONE


class FormulaNode:
    kind: FormulaKind

    def evalueate(self):
        # TODO high priority
        # this is the logic from datamancer, once enough logic is in place
        # this has to be ported
        # https://github.com/SciNim/Datamancer/blob/47ba4d81bf240a7755b73bc48c1cec9b638d18ae/src/datamancer/dataframe.nim#L2515
        return VString("TODO")
