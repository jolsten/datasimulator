import itertools
import numpy as np
from dataclasses import dataclass
from typing import Iterable
from .typing import DTypeLike, Number, DateTime


def _validate_number(value, dtype) -> None:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
    else:
        return
    if value < info.min or value > info.max:
        raise ValueError(
            f'Value w/ dtype="{dtype}" must be in range'
            f' [{info.min}, {info.max}] but is {value}'
        )


@dataclass
class Constant:
    value: Number = 0
    dtype: DTypeLike = 'u1'

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        _validate_number(self.value, self.dtype)

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(self.value, dtype=self.dtype)


@dataclass
class Counter:
    start: int = 0
    step: int = 1
    dtype: DTypeLike = 'u1'

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        _validate_number(self.start, self.dtype)
        self.iterator = itertools.count(self.start, self.step)

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        modulo = np.iinfo(self.dtype).max + 1
        return np.array(next(self.iterator) % modulo, dtype=self.dtype)


@dataclass
class Cycle:
    values: Iterable[Number]
    dtype: DTypeLike = 'u1'

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        [_validate_number(v, self.dtype) for v in self.values]
        self.iterator = itertools.cycle(self.values)

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(next(self.iterator), dtype=self.dtype)


@dataclass
class Clock:
    epoch: DateTime
    dtype: DTypeLike = 'u4'

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array((t - self.epoch).total_seconds(), dtype=self.dtype)


@dataclass
class TimeFunction:
    func: callable
    dtype: DTypeLike = 'u1'

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
    
    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(self.func(t), dtype=self.dtype)


@dataclass
class IndexFunction:
    func: callable
    dtype: DTypeLike = 'u1'

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
    
    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(self.func(x), dtype=self.dtype)
