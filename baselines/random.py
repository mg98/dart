import numpy as np
from utils.common import ranking_func

@ranking_func
def random_rank(_, activities):
    for ua in activities:
        np.random.shuffle(ua.results)
    return activities