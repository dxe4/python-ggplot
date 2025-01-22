from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_ggplot.gg_types import GGValue


class ColumnType(Enum):
    NONE = auto()
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    STRING = auto()
    OBJECT = auto()
    CONSTANT = auto()
    GENERIC = auto()


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


# Type alias for all possible column types
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


FormulaNode = Formula[Column]
