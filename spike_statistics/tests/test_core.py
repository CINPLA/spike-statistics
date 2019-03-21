import pytest
import numpy as np
from spike_statistics.process import stationary_poisson


def test_fano():
    #The fano factor is 1 for Poisson processes
    from spike_statistics.core import fano_factor
    np.random.seed(12345)
    t1 = [stationary_poisson(0.0, 1, 1) for _ in range(1000)]
    assert abs(fano_factor(t1) - 1) < .02
