import numpy as np
import pytest
import datasimulator as ds

@pytest.mark.parametrize('value, dtype', [
    (  0, 'u1'),
    ( 85, 'u1'),
    (170, 'u1'),
    (255, 'u1'),
])
def test_constant(value, dtype):
    m = ds.Constant(value=value, dtype=dtype)
    for _ in range(10):
        print(m.generate(0,0), value)
        assert m.generate(0, 0) == value

def test_counter():
    m = ds.Counter(dtype='u1')
    for i in range(10):
        assert m.generate(0, 0) == i

@pytest.mark.parametrize('value_list, dtype', [
    (range(4), 'u1'),
    (range(8), 'u1'),
])
def test_cycle(value_list, dtype):
    value_list = list(value_list)
    m = ds.Cycle(value_list, dtype=dtype)
    for v in value_list + value_list + value_list:
        assert m.generate(0, 0) == v
