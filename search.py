import numpy as np
import matplotlib.pyplot as plt
from world import GRID, make_prob_map, sample_target, hotspots

SENSE_RADIUS = 2  # drone can see 2 cells in every direction

def can_see(drone, target):
    return (abs(drone[0] - target[0]) <= SENSE_RADIUS and
            abs(drone[1] - target[1]) <= SENSE_RADIUS)

def lawnmower_path(size=GRID):
    spacing = 2 * SENSE_RADIUS + 1   # rows far enough apart to not overlap
    path = []
    r = SENSE_RADIUS
    going_right = True
    while r < size:
        cols = range(size) if going_right else range(size - 1, -1, -1)
        for c in cols:
            path.append((r, c))
        end_col = path[-1][1]
        next_r = r + spacing
        for step_r in range(r + 1, min(next_r, size)):  # fly up to next row
            path.append((step_r, end_col))
        r = next_r
        going_right = not going_right
    return path

def run_search(path, target):
    for step, pos in enumerate(path, start=1):
        if can_see(pos, target):
            return step, path[:step]
    return None, path

if __name__ == "__main__":
    prob = make_prob_map(hotspots)
    path = lawnmower_path()

    # One trial, visualized
    target = sample_target(prob)
    steps, flown = run_search(path, target)
    rows = [p[0] for p in flown]
    cols = [p[1] for p in flown]
    plt.imshow(prob, origin="lower", cmap="hot")
    plt.plot(cols, rows, c="white", linewidth=1)
    plt.scatter(target[1], target[0], c="cyan", marker="x", s=100)
    plt.title(f"Lawnmower: found in {steps} steps")
    plt.show()

    # 100 trials
    results = []
    for trial in range(100):
        t = sample_target(prob)
        s, _flown = run_search(path, t)
        results.append(s)
    print(f"Lawnmower average over 100 trials: {np.mean(results):.1f} steps")