import numpy as np


# Sample from a range of values that have an *absolute* value
# in the specified range with uniform probability. This implies
# sampled values can be positive or negative.
def _non_zero_uniform_sampler(abs_low, abs_high, size):
    assert(0 < abs_low < abs_high)

    vals = np.random.uniform(low=abs_low, high=abs_high, size=size)
    neg_locs = (np.random.random(size=size) < 0.5)
    neg_mask = np.full(size, 1)
    neg_mask[neg_locs] = -1
    return vals*neg_mask
