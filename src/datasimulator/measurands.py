import itertools
import numpy as np
from dataclasses import dataclass
from typing import Iterable
from .typing import DTypeLike, Number, DateTime


@dataclass
class Constant:
    value: Number = 0
    dtype: DTypeLike = 'u1'

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(self.value, dtype=self.dtype)


@dataclass
class Counter:
    start: int = 0
    step: int = 1
    dtype: DTypeLike = 'u1'

    def __post_init__(self) -> None:
        self.iterator = itertools.count(self.start, self.step)

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(next(self.iterator), dtype=self.dtype)


@dataclass
class Cycle:
    values: Iterable[Number]
    dtype: DTypeLike = 'u1'

    def __post_init__(self) -> None:
        self.iterator = itertools.cycle(self.values)

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(next(self.iterator), dtype=self.dtype)


@dataclass
class Clock:
    epoch: DateTime
    dtype: DTypeLike = 'u4'

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array((t - self.epoch).total_seconds(), dtype=self.dtype)


@dataclass
class TimeFunction:
    func: callable
    dtype: DTypeLike = 'u1'
    
    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(self.func(t), dtype=self.dtype)


@dataclass
class IndexFunction:
    func: callable
    dtype: DTypeLike = 'u1'
    
    def generate(self, x: int, t: DateTime) -> np.ndarray:
        return np.array(self.func(x), dtype=self.dtype)
