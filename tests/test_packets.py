import numpy as np
import pytest
from hypothesis import given, assume, strategies as st
import datasimulator as ds

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
