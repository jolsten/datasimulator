import itertools
import numpy as np
from dataclasses import dataclass
from typing import Iterable
from .typing import DateTime, DataSource


@dataclass
class Frame:
    sources: Iterable[DataSource]
    msbf: bool = True

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        # print('aa', self.sources)
        data = [np.atleast_1d(s.generate(x, t)) for s in self.sources]
        if self.msbf:
            data = [d.byteswap() for d in data]
        data = [d.view('u1') for d in data]
        return np.concatenate(data)


@dataclass
class CycleFrame:
    frames: Iterable[Frame]
    msbf: bool = True

    def __post_init__(self) -> None:
        self._index = itertools.cycle(range(len(self.frames)))

        # Frames must be the same length
        lengths = [len(a.sources) for a in self.frames]
        if not all(a == lengths[0] for a in lengths):
            raise ValueError(f'Frames must be the same length, but are: {lengths}')

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        idx = next(self._index)
        current_frame = self.frames[idx % len(self.frames)]
        data = current_frame.generate(x, t)
        return np.concatenate([np.array([idx], dtype='u1'), data])
