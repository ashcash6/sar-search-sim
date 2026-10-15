import numpy as np
import matplotlib.pyplot as plt
from world import GRID, make_prob_map, sample_target, hotspots
from search import SENSE_RADIUS, can_see, lawnmower_path, run_search

K = 10  # number of frequencies in each direction

# Fourier basis: one cosine wave pattern per (k1, k2) pair, evaluated at every cell
k1, k2 = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
cols, rows = np.meshgrid(np.arange(GRID) + 0.5, np.arange(GRID) + 0.5)
BASIS = (np.cos(k1[:, :, None, None] * np.pi * cols / GRID) *
         np.cos(k2[:, :, None, None] * np.pi * rows / GRID))  # shape (K, K, GRID, GRID)

# Weights: low frequencies matter more
LAMBDA = (1 + k1**2 + k2**2) ** -1.5

def target_coeffs(prob):
    return (BASIS * prob).sum(axis=(2, 3))

def ergodic_metric(c, phi):
    return np.sum(LAMBDA * (c - phi) ** 2)

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def ergodic_search(prob, target, start=(SENSE_RADIUS, 0), max_steps=3000, update_map=False):
    prob = prob.copy()
    phi = target_coeffs(prob)
    pos = start
    visit_sum = BASIS[:, :, pos[0], pos[1]].copy()
    path = [pos]
    R = SENSE_RADIUS
    for step in range(1, max_steps + 1):
        if can_see(pos, target):
            return step, path

        if update_map:
            # Cross off the area we just looked at: person isn't there
            r0, r1 = max(0, pos[0] - R), min(GRID, pos[0] + R + 1)
            c0, c1 = max(0, pos[1] - R), min(GRID, pos[1] + R + 1)
            seen = prob[r0:r1, c0:c1]
            removed = seen.sum()
            if 0 < removed < 1:
                phi -= (BASIS[:, :, r0:r1, c0:c1] * seen).sum(axis=(2, 3))
                prob[r0:r1, c0:c1] = 0
                prob /= (1 - removed)
                phi /= (1 - removed)

        best_move, best_score = None, np.inf
        for dr, dc in MOVES:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < GRID and 0 <= nc < GRID:
                c = (visit_sum + BASIS[:, :, nr, nc]) / (step + 1)
                score = ergodic_metric(c, phi)
                if score < best_score:
                    best_move, best_score = (nr, nc), score
        pos = best_move
        visit_sum += BASIS[:, :, pos[0], pos[1]]
        path.append(pos)
    return None, path

if __name__ == "__main__":
    prob = make_prob_map(hotspots)
    lawn = lawnmower_path()
    targets = [sample_target(prob) for _ in range(100)]

    
    lawn_steps, erg_steps, upd_steps = [], [], []
    for t in targets:
        s_lawn, _ = run_search(lawn, t)
        s_erg, _ = ergodic_search(prob, t)
        s_upd, _ = ergodic_search(prob, t, update_map=True)
        lawn_steps.append(s_lawn)
        erg_steps.append(s_erg if s_erg is not None else 3000)
        upd_steps.append(s_upd if s_upd is not None else 3000)

    print(f"Lawnmower avg:          {np.mean(lawn_steps):.1f} steps")
    print(f"Ergodic avg:            {np.mean(erg_steps):.1f} steps")
    print(f"Ergodic + updates avg:  {np.mean(upd_steps):.1f} steps")

  

    # Visualize one ergodic run
    t = targets[0]
    steps, path = ergodic_search(prob, t)
    plt.imshow(prob, origin="lower", cmap="hot")
    plt.plot([p[1] for p in path], [p[0] for p in path], c="white", linewidth=1)
    plt.scatter(t[1], t[0], c="cyan", marker="x", s=100)
    plt.title(f"Ergodic: found in {steps} steps")
    plt.show()