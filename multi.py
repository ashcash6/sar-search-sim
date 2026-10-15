import numpy as np
import matplotlib.pyplot as plt
from world import GRID, make_prob_map, sample_target
from search import SENSE_RADIUS, can_see
from ergodic import BASIS, target_coeffs, ergodic_metric, MOVES
from experiments import random_hotspots

def cross_off(prob, phi, pos):
    """Bayesian update in place: the area this drone sees is empty."""
    R = SENSE_RADIUS
    r0, r1 = max(0, pos[0] - R), min(GRID, pos[0] + R + 1)
    c0, c1 = max(0, pos[1] - R), min(GRID, pos[1] + R + 1)
    seen = prob[r0:r1, c0:c1]
    removed = seen.sum()
    if 0 < removed < 1:
        phi -= (BASIS[:, :, r0:r1, c0:c1] * seen).sum(axis=(2, 3))
        prob[r0:r1, c0:c1] = 0
        prob /= (1 - removed)
        phi /= (1 - removed)

def best_move(pos, visit_sum, count, phi):
    best, best_score = pos, np.inf
    for dr, dc in MOVES:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            c = (visit_sum + BASIS[:, :, nr, nc]) / (count + 1)
            score = ergodic_metric(c, phi)
            if score < best_score:
                best, best_score = (nr, nc), score
    return best

def multi_search(prob0, target, starts, shared, max_steps=1500):
    n = len(starts)
    positions = list(starts)
    paths = [[p] for p in positions]

    if shared:
        prob = prob0.copy()
        phi = target_coeffs(prob)
        visit = sum(BASIS[:, :, p[0], p[1]] for p in positions)
        count = n
    else:
        probs = [prob0.copy() for _ in range(n)]
        phis = [target_coeffs(prob0) for _ in range(n)]
        visits = [BASIS[:, :, p[0], p[1]].copy() for p in positions]
        counts = [1] * n

    for step in range(1, max_steps + 1):
        if any(can_see(p, target) for p in positions):
            return step, paths
        for i in range(n):
            if shared:
                cross_off(prob, phi, positions[i])
                positions[i] = best_move(positions[i], visit, count, phi)
                visit += BASIS[:, :, positions[i][0], positions[i][1]]
                count += 1
            else:
                cross_off(probs[i], phis[i], positions[i])
                positions[i] = best_move(positions[i], visits[i], counts[i], phis[i])
                visits[i] += BASIS[:, :, positions[i][0], positions[i][1]]
                counts[i] += 1
            paths[i].append(positions[i])
    return max_steps, paths

def sector_lawnmower(c0, c1):
    """Lawnmower sweep restricted to columns c0 to c1 - 1."""
    spacing = 2 * SENSE_RADIUS + 1
    path = []
    r = SENSE_RADIUS
    going_right = True
    while r < GRID:
        cols = range(c0, c1) if going_right else range(c1 - 1, c0 - 1, -1)
        for c in cols:
            path.append((r, c))
        end_col = path[-1][1]
        for step_r in range(r + 1, min(r + spacing, GRID)):
            path.append((step_r, end_col))
        r += spacing
        going_right = not going_right
    return path

def parallel_search(target, paths, max_steps=1500):
    longest = max(len(p) for p in paths)
    for step in range(1, min(longest, max_steps) + 1):
        positions = [p[min(step - 1, len(p) - 1)] for p in paths]
        if any(can_see(pos, target) for pos in positions):
            return step, [p[:step] for p in paths]
    return max_steps, paths

if __name__ == "__main__":
    import os
    os.makedirs("multi_maps", exist_ok=True)

    starts = [(SENSE_RADIUS, 0), (SENSE_RADIUS, GRID // 2), (SENSE_RADIUS, GRID - 1)]
    sectors = [(0, 17), (17, 34), (34, GRID)]
    par_paths = [sector_lawnmower(c0, c1) for c0, c1 in sectors]

    N_MAPS, N_TRIALS = 10, 30
    colors = ["white", "cyan", "lime"]
    shared_all, indep_all, par_all = [], [], []
    wins_comm, wins_par = 0, 0

    for m in range(N_MAPS):
        prob = make_prob_map(random_hotspots())
        targets = [sample_target(prob) for _ in range(N_TRIALS)]
        s = np.mean([multi_search(prob, t, starts, shared=True)[0] for t in targets])
        i = np.mean([multi_search(prob, t, starts, shared=False)[0] for t in targets])
        p = np.mean([parallel_search(t, par_paths)[0] for t in targets])
        shared_all.append(s); indep_all.append(i); par_all.append(p)
        wins_comm += s < i
        wins_par += s < p
        print(f"Map {m+1:2d}:  shared {s:6.1f}   no comm {i:6.1f}   parallel lawnmower {p:6.1f}")

        # Save a picture of this map (first target), all three setups
        t0 = targets[0]
        runs = [
            ("Shared ergodic", multi_search(prob, t0, starts, shared=True)),
            ("No-comm ergodic", multi_search(prob, t0, starts, shared=False)),
            ("Parallel lawnmower", parallel_search(t0, par_paths)),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (name, (steps, paths)) in zip(axes, runs):
            ax.imshow(prob, origin="lower", cmap="hot")
            for path, col in zip(paths, colors):
                ax.plot([q[1] for q in path], [q[0] for q in path], c=col, linewidth=1)
            ax.scatter(t0[1], t0[0], c="red", marker="x", s=120)
            ax.set_title(f"{name}: {steps} steps")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Map {m+1}  (avg over {N_TRIALS} trials: shared {s:.1f}, "
                     f"no comm {i:.1f}, parallel {p:.1f})")
        plt.tight_layout()
        plt.savefig(f"multi_maps/map_{m+1:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print()
    print(f"Overall avg:  shared {np.mean(shared_all):.1f}   no comm {np.mean(indep_all):.1f}   "
          f"parallel {np.mean(par_all):.1f}")
    print(f"Shared beat no-comm on {wins_comm}/{N_MAPS} maps")
    print(f"Shared beat parallel lawnmower on {wins_par}/{N_MAPS} maps")
    print("Saved one image per map in multi_maps/")