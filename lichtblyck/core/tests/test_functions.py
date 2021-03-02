from lichtblyck.core import functions
import numpy as np
import pytest


freqs_small_to_large = ['T', '5T', '15T', '30T', 'H', '2H', 'D', 'MS', 'QS', 'AS']


@pytest.fixture(params=freqs_small_to_large)
def freq1(request):
    return request.param

@pytest.fixture(params=freqs_small_to_large)
def freq2(request):
    return request.param


def test_freq_diff(freq1, freq2):
    i1 = freqs_small_to_large.index(freq1)
    i2 = freqs_small_to_large.index(freq2)
    outcome = np.sign(i1 - i2)
    assert functions.freq_diff(freq1, freq2) == outcome

