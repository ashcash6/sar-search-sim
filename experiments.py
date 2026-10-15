import numpy as np
from world import GRID, make_prob_map, sample_target
from search import lawnmower_path, run_search
from ergodic import ergodic_search

rng = np.random.default_rng(42)

def random_hotspots(n=3):
    spots = []
    for _ in range(n):
        cx, cy = rng.integers(5, GRID - 5, size=2)
        spread = rng.uniform(2, 7)
        weight = rng.uniform(0.3, 1.0)
        spots.append((cx, cy, spread, weight))
    return spots

if __name__ == "__main__":
    lawn = lawnmower_path()
    N_MAPS, N_TRIALS = 20, 50
    all_l, all_e, all_u = [], [], []
    wins = 0

    for m in range(N_MAPS):
        prob = make_prob_map(random_hotspots())
        targets = [sample_target(prob) for _ in range(N_TRIALS)]

        l = np.mean([run_search(lawn, t)[0] for t in targets])
        e = np.mean([ergodic_search(prob, t)[0] or 3000 for t in targets])
        u = np.mean([ergodic_search(prob, t, update_map=True)[0] or 3000 for t in targets])

        all_l.append(l); all_e.append(e); all_u.append(u)
        if u < l:
            wins += 1
        print(f"Map {m+1:2d}:  lawn {l:6.1f}   ergodic {e:6.1f}   ergodic+upd {u:6.1f}")

    print()
    print(f"Overall avg:  lawn {np.mean(all_l):.1f}   ergodic {np.mean(all_e):.1f}   ergodic+upd {np.mean(all_u):.1f}")
    print(f"Ergodic + updates beat lawnmower on {wins}/{N_MAPS} maps")