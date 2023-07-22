import datetime
import numpy as np
from dataclasses import dataclass
from numpy.typing import DTypeLike
from typing import Protocol, Union

Number = Union[int, float]
DateTime = datetime.datetime

class DataSource(Protocol):
    def generate(self, x: int, t: datetime.datetime) -> np.ndarray:
        ...
