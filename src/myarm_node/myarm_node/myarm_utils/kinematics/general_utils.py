import numpy as np
from scipy.stats import qmc


# generate random samples following a defined random distribution
def generate_random_samples(lb, ub, sample_size = 100, sampler = "normal", seed = None):
    rng = np.random.default_rng(seed = seed)
    lb = np.copy(lb.flatten())
    ub = np.copy(ub.flatten())
    if sampler == "normal":
        mean = (lb + ub) / 2
        std = (ub - lb) / 4
        samples = np.array([rng.normal(loc = mean, scale = std) for _ in range(sample_size)])
        samples = np.clip(samples, lb, ub)
    elif sampler == "uniform":
        samples = np.array([rng.uniform(lb, ub) for _ in range(sample_size)])
    elif sampler == "sobol":
        if not np.log2(sample_size).is_integer():
            sample_size = int(2.0 ** round(np.log2(sample_size)))
        sampler = qmc.Sobol(d = len(lb), scramble = True, seed = seed)
        unit_samples = sampler.random(sample_size)
        samples = qmc.scale(unit_samples, lb, ub)
    elif sampler == "halton":
        if not np.log2(sample_size).is_integer():
            sample_size = int(2.0 ** round(np.log2(sample_size)))
        sampler = qmc.Halton(d = len(lb), scramble = True)
        unit_samples = sampler.random(sample_size)
        samples = qmc.scale(unit_samples, lb, ub)
    return samples
