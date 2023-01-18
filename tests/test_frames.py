import numpy as np
import pytest
import datasimulator as ds

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
