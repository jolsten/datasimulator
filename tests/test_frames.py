import numpy as np
import pytest
from hypothesis import given, assume, strategies as st
from typing import Optional, Iterable

import datasimulator as ds

def numpy_integers(dtype='u1'):
    info = np.iinfo(dtype)
    return st.integers(min_value=info.min, max_value=info.max)

@st.composite
def ex_constant(draw):
    dtype = draw(st.sampled_from(['u1', 'u2', 'u4', 'i1', 'i2', 'i4']))
    value = numpy_integers(dtype)
    return st.builds(ds.Constant, value=value, dtype=dtype)

@st.composite
def ex_counter(draw):
    dtype = draw(st.sampled_from(['u1', 'u2', 'u4']))
    start = draw(numpy_integers(dtype))
    step = draw(numpy_integers(dtype))
    return st.builds(ds.Counter, start=start, step=step, dtype=dtype)

def ex_source(strategies=[ex_constant(), ex_counter()]):
    return st.one_of(strategies)

def ex_frame(min_size=1, max_size=512):
    source_list = st.lists(
        ex_source(),
        min_size=min_size,
        max_size=max_size,
    )
    return st.builds(ds.Frame, sources=source_list)

@given(ex_frame())
def test_frame(frame: ds.Frame):
    print(frame)
    data = frame.generate(0, 0)
    assert False


@pytest.mark.parametrize('length', (4, 16, 32, 64, 128, 256, 512, 1024))
def test_frame_constant(length):
    measurands = [ds.Constant(v % 256, dtype='u1') for v in range(length)]
    frame = ds.Frame(measurands)
    data = frame.generate(0, 0)

    assert len(data) == length
    for i in range(length):
        assert data[i] == i % 256


def test_frame_lengths_nomatch():
    with pytest.raises(ValueError):
        cf = ds.CycleFrame([
            ds.Frame([ds.Constant(v % 256, dtype='u1') for v in range(10)]),
            ds.Frame([ds.Constant(v % 256, dtype='u1') for v in range(11)]),
        ])


def test_cycle_frame():
    frames = [ds.Frame([ds.Constant(i, dtype='u1') for _ in range(48)]) for i in range(4)]
    cycle_frame = ds.CycleFrame(frames)

    for i in range(12):
        data = cycle_frame.generate(0, 0)
        assert data[0] == i % 4


def create_generic_packet(apid: int) -> ds.Packet:
    pdl = (apid * 7 + 3) % 128
    return ds.Packet(apid=apid, sources=[ds.Constant(apid) for _ in range(pdl)])


@pytest.mark.parametrize('pid', [1, 5, 10, 25, 50, 100, 150, 200])
def test_packet_pid(pid):
    p = ds.Packet(
        apid=pid,
        sources=[ds.Constant(pid) for _ in range(pid+1)],
    )
    data = p.generate(0, 0)

    vals, counts = np.unique(data[6:], return_counts=True)
    mode = vals[np.argmax(counts)]
    assert mode == pid


def test_packet_pid_too_large():
    with pytest.raises(ValueError):
        p = ds.Packet(
            apid=65536,
            sources=[ds.Constant(1)]
        )


def test_packet_sequence_number():
    p = ds.Packet(
        apid=85,
        sources=[ds.Constant(0, dtype='u4')],
    )
    for idx in range(10):
        data = p.generate(0,0)
        seqn = data[0:6].view('u2').byteswap()[1]
        assert seqn == idx


@pytest.mark.parametrize('pdl', [1, 10, 50, 100])
def test_packet_length(pdl):
    p = ds.Packet(
        apid=85,
        sources=[ds.Constant(0) for _ in range(pdl)],
    )
    data = p.generate(0,0)
    assert len(data) == pdl + 6


# def test_packet_data_unit():
#     pdu = ds.PacketDataUnit(
#         size=50,
#         packets=[
#             ds.Packet(5, sources=[ds.Constant(5) for _ in range(15)]),
#         ],
#         cycle=[5],
#     )

#     data = pdu.generate(0, 0)
#     print(data, data.shape)
#     assert False
