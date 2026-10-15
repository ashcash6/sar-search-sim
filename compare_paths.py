import matplotlib.pyplot as plt
from world import make_prob_map, sample_target, hotspots
from search import lawnmower_path, run_search
from ergodic import ergodic_search

prob = make_prob_map(hotspots)
target = sample_target(prob)

runs = [
    ("Lawnmower", run_search(lawnmower_path(), target)),
    ("Ergodic", ergodic_search(prob, target)),
    ("Ergodic + updates", ergodic_search(prob, target, update_map=True)),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, (steps, path)) in zip(axes, runs):
    ax.imshow(prob, origin="lower", cmap="hot")
    ax.plot([p[1] for p in path], [p[0] for p in path], c="white", linewidth=1)
    ax.scatter(target[1], target[0], c="cyan", marker="x", s=100)
    ax.set_title(f"{name}: {steps} steps")
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("path_comparison.png", dpi=150, bbox_inches="tight")
plt.show()