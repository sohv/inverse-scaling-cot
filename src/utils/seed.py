import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set random seed for random and numpy.

    Does NOT set torch seed -- vLLM handles its own seeding via SamplingParams.seed.
    """
    random.seed(seed)
    np.random.seed(seed)
