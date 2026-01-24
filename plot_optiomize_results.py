import glob
import os
import optuna
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

db_files = glob.glob("*.db")

records = []

for db in db_files:
    storage = f"sqlite:///{os.path.abspath(db)}"
    summaries = optuna.get_all_study_summaries(storage)

    if not summaries:
        continue

    study = optuna.load_study(
        study_name=summaries[0].study_name,
        storage=storage,
    )

    best_value = study.best_value
    if best_value == -1:
        continue

    # first trial value (if it exists)
    first_value = (
        study.trials[0].value
        if study.trials and study.trials[0].value is not None
        else None
    )

    valid_trials = [
        t for t in study.trials
        if t.value is not None and t.value != -1
    ]

    records.append({
        "name": os.path.basename(db),
        "best": best_value,
        "first": first_value,
        "trials": len(valid_trials),
    })

# ---- Sort and rank ----
records.sort(key=lambda r: r["best"], reverse=True)
for i, r in enumerate(records, start=1):
    r["rank"] = i

names = [r["name"] for r in records]
best_scores = [r["best"] for r in records]
first_scores = [r["first"] for r in records]
num_trials = [r["trials"] for r in records]
ranks = [r["rank"] for r in records]

# ---- Color map by rank ----
norm = np.linspace(1, 0, len(records))
colors = cm.viridis(norm)

# ---- Plot: Max score (horizontal, spacious) ----
plt.figure(figsize=(10, 0.5 * len(records) + 2))
bars = plt.barh(names, best_scores, color=colors)
plt.xlabel("Score")
plt.title("Best Optuna score per DB")

for bar, best, first in zip(bars, best_scores, first_scores):
    label = f"max: {best:.4g} | first: "
    label += f"{first:.4g}" if first is not None else "n/a"

    plt.text(
        bar.get_width() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        label,
        va="center",
        ha="left",
        fontsize=9,
        color="white" if bar.get_width() > max(best_scores) * 0.15 else "black",
    )

plt.tight_layout()

# ---- Plot: Valid trial count (horizontal) ----
plt.figure(figsize=(10, 0.5 * len(records) + 2))
bars = plt.barh(names, num_trials, color=colors)
plt.xlabel("Trials (value != -1)")
plt.title("Number of valid trials per DB")

for bar, trials, rank in zip(bars, num_trials, ranks):
    plt.text(
        bar.get_width() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{trials}  |  #{rank}",
        va="center",
        ha="left",
        fontsize=9,
    )

plt.tight_layout()
plt.show()
