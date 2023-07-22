import datetime
import numpy as np
from hypothesis import given, assume, strategies as st
import datasimulator as ds
from .strategies import ex_constant, ex_counter, ex_cycle, ex_clock, measurands


@given(ex_constant())
def test_constant(m: ds.Constant):
    for _ in range(10):
        sample = m.generate(0,0)
        assert sample == m.value

@given(ex_counter())
def test_counter(m: ds.Counter):
    start = m.start
    step = m.step
    dtype = m.dtype
    for i in range(10):
        expected_value = (start + i * step) % (np.iinfo(dtype).max + 1)
        assert m.generate(0, 0) == expected_value

@given(ex_cycle())
def test_cycle_u1(m: ds.Cycle):
    value_list = m.values
    for v in value_list + value_list + value_list:
        assert m.generate(0, 0) == v

@given(ex_clock(), st.integers(min_value=1, max_value=1000))
def test_clock(m: ds.Clock, dt: int):
    dt = datetime.timedelta(seconds=dt)
    for i in range(100):
        c = m.epoch + dt*i
        value = m.generate(0, m.epoch+dt*i).view('u4')
        print(type(value), value)
        assert value == (c - m.epoch).total_seconds()

@given(measurands())
def test_measurands(m):
    start = datetime.datetime.now()
    for i in range(10):
        t = start + datetime.timedelta(seconds=100*i)
        assert isinstance(m.generate(i, t), np.ndarray)
