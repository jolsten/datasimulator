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
