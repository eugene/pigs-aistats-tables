import numpy as np
from scipy.stats import norm

def test_mean_difference(dist_1, dist_2, level=0.05):
    n = norm(
        loc=0,
        scale=np.sqrt(
            (dist_1["std"] ** 2 / dist_1["n"]) + (dist_2["std"] ** 2 / dist_2["n"])
        ),
    )
    p_value = 2 * n.sf(abs(dist_1["mean"] - dist_2["mean"]))
    # Is there a statistical difference?
    return p_value < level

if __name__ == '__main__':
    dist_1 = {
        "mean": 0.55,
        "std": 0.03,
        "n": 20,
    }

    dist_2 = {
        "mean": 0.57,
        "std": 0.02,
        "n": 20,
    }

    print(test_mean_difference(dist_1, dist_2))
