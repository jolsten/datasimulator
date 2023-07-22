import datetime
from typing import Optional, Iterable
import numpy as np
from hypothesis import given, assume, strategies as st
import datasimulator as ds

@st.composite
def numpy_integers(draw, dtype='u1'):
    info = np.iinfo(dtype)
    return draw(st.integers(min_value=info.min, max_value=info.max))

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

@st.composite
def ex_constant(draw):
    dtype = draw(st.sampled_from(['u1', 'u2', 'u4', 'i1', 'i2', 'i4']))
    return draw(st.builds(
        ds.Constant,
        value=numpy_integers(dtype),
        dtype=st.just(dtype)
    ))

@st.composite
def ex_counter(draw):
    dtype = draw(st.sampled_from(['u1', 'u2', 'u4']))
    return draw(st.builds(
        ds.Counter,
        start=numpy_integers(dtype),
        step=numpy_integers(dtype),
        dtype=st.just(dtype)
    ))

@st.composite
def ex_cycle(draw, strategy=numpy_integers('u1'), min_size=1, max_size=64):
    return draw(st.builds(
        ds.Cycle,
        st.lists(strategy, min_size=min_size, max_size=max_size)
    ))

MIN_DATETIME = datetime.datetime(1970,1,1)
MAX_DATETIME = datetime.datetime(2050,1,1)

@st.composite
def ex_clock(draw):
    return draw(st.builds(
        ds.Clock,
        epoch=st.datetimes(
            min_value=MIN_DATETIME,
            max_value=MAX_DATETIME,
            timezones=st.just(None),
            allow_imaginary=False
        )
    ))


@st.composite
def simple_measurands(draw, strategies=[ex_constant(), ex_counter()]):
    return draw(st.one_of(strategies))

@st.composite
def measurands(draw, options=[ex_constant(), ex_counter(), ex_cycle(), ex_clock()]):
    return draw(st.one_of(options))
