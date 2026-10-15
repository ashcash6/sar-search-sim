import numpy as np
import matplotlib.pyplot as plt

GRID = 50
rng = np.random.default_rng(0)

def make_prob_map(hotspots, size=GRID):
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    prob = np.zeros((size, size))
    for (cx, cy, spread, weight) in hotspots:
        prob += weight * np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * spread**2))
    return prob / prob.sum()

hotspots = [(12, 15, 4, 1.0), (35, 30, 6, 0.7), (20, 40, 3, 0.5)]
prob = make_prob_map(hotspots)

def sample_target(prob):
    flat_index = rng.choice(prob.size, p=prob.ravel())
    return np.unravel_index(flat_index, prob.shape)  # (row, col)

if __name__ == "__main__":

    target = sample_target(prob)

    plt.imshow(prob, origin="lower", cmap="hot")
    plt.scatter(target[1], target[0], c="cyan", marker="x", s=100, label="person")
    plt.colorbar(label="probability")
    plt.legend()
    plt.title("Where the missing person might be")
    plt.show()