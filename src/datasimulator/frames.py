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
        print('aa', self.sources)
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


@dataclass
class Packet:
    apid: int
    sources: Iterable[DataSource]

    def __post_init__(self) -> None:
        if self.apid > 2**11:
            raise ValueError(f'Packet ID value is {self.apid} but must be < {2**11}')
        
        self._sequence_iter = itertools.count()

    def _make_header(self, pid: int, pdl: int) -> np.ndarray:
        seqn = next(self._sequence_iter)
        return np.array([pid, seqn, pdl], 'u2').byteswap().view('u1')
    
    def generate(self, x: int, t: DateTime) -> np.ndarray:
        data = np.concatenate(
            [np.atleast_1d(s.generate(x, t)).byteswap().view('u1') for s in self.sources]
        )
        hdr = self._make_header(self.apid, len(data)-1)
        return np.concatenate([hdr, data])


@dataclass
class PacketDataUnit:
    size: int
    packets: Iterable[Packet]
    cycle: Iterable[int]

    def __post_init__(self) -> None:
        self._iterator = itertools.cycle(self.cycle)
        self._dict = {p.apid: p for p in self.packets}
        self._spillover = np.array([], dtype='u1')

    def generate(self, x: int, t: DateTime) -> np.ndarray:
        fpp = len(self._spillover)
        current = [
            np.array([fpp], dtype='u2').byteswap().view('u1'),
            self._spillover,
        ]
        while sum([len(x) for x in current]) < self.size:
            next_apid = next(self._iterator)
            next_packet = self._dict[next_apid].generate(x, t)
            current.append(next_packet)

        data = np.concatenate(current)
        self._spillover = data[self.size:]
        return data[0:self.size]
