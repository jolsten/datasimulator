import datetime
from hypothesis import given, assume, strategies as st
import numpy as np
import pytest
from typing import Optional, Iterable

import datasimulator as ds


@st.composite
def int_of_dtype(draw, dtype='u1'):
    info = np.iinfo(np.dtype(dtype))
    return draw(st.integers(min_value=info.min, max_value=info.max))


@st.composite
def int_and_dtype(draw, signed: Optional[bool]=None, sizes: Iterable[int] = [1,2,4,8]):
    if signed is None:
        signs = ['u', 'i']
    elif signed:
        signs = ['i']
    else:
        signs = ['u']
    sign = draw(st.sampled_from(signs))
    size = draw(st.sampled_from(sizes))
    dtype = f'{sign}{size}'
    value = draw(int_of_dtype(dtype))
    return value, dtype


@given(int_and_dtype())
def test_constant(value_and_dtype):
    value, dtype = value_and_dtype
    m = ds.Constant(value=value, dtype=dtype)
    for _ in range(10):
        sample = m.generate(0,0)
        assert sample == value


@given(int_and_dtype(signed=False), st.integers(min_value=-10, max_value=10))
def test_counter(value_and_dtype, step):
    value, dtype = value_and_dtype
    m = ds.Counter(start=value, step=step, dtype=dtype)
    for i in range(10):
        expected_value = (value + i * step) % (np.iinfo(dtype).max + 1)
        assert m.generate(0, 0) == expected_value


@given(st.lists(st.integers(min_value=0, max_value=255)))
def test_cycle_u1(value_list):
    value_list = list(value_list)
    m = ds.Cycle(value_list, dtype='u1')
    for v in value_list + value_list + value_list:
        assert m.generate(0, 0) == v


_dt_kwargs = dict(
    min_value=datetime.datetime(2000,1,1),
    max_value=datetime.datetime(2030,1,1),
    timezones=st.sampled_from([None]),
    allow_imaginary=False,
)
@given(st.datetimes(**_dt_kwargs), st.datetimes(**_dt_kwargs))
def test_clock(epoch, dt):
    value = int((dt - epoch).total_seconds())
    assume(epoch < dt)
    assume(value < 2**32)
    m = ds.Clock(epoch=epoch)
    assert m.generate(0, dt) == value
