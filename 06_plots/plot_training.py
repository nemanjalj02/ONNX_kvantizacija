import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

THIS_DIR = Path(__file__).resolve().parent
OUT_PATH = THIS_DIR / "training_curve.png"

epochs = list(range(1, 15))
train_loss = [0.2382, 0.0672, 0.0458, 0.0355, 0.0296, 0.0250, 0.0221,
              0.0174, 0.0161, 0.0150, 0.0116, 0.0132, 0.0102, 0.0086]
val_acc = [0.9698, 0.9806, 0.9856, 0.9860, 0.9866, 0.9836, 0.9872,
           0.9850, 0.9878, 0.9868, 0.9884, 0.9882, 0.9868, 0.9840]

best_epoch = 11
stop_epoch = 14

fig, ax1 = plt.subplots(figsize=(5, 3.2))

ax1.set_xlabel("Epoha")
ax1.set_ylabel("Gubitak (trening)", color="tab:red")
ax1.plot(epochs, train_loss, "o-", color="tab:red", label="Gubitak (trening)")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("Tačnost (validacija)", color="tab:blue")
ax2.plot(epochs, val_acc, "o-", color="tab:blue", label="Tačnost (val.)")
ax2.tick_params(axis="y", labelcolor="tab:blue")

ax2.scatter([best_epoch], [val_acc[best_epoch - 1]], s=140, facecolors="none",
            edgecolors="green", linewidths=2, zorder=5, label="najbolja epoha")

ax1.axvline(stop_epoch, color="gray", linestyle="--", linewidth=1)

plt.title("Trening LeNet5: gubitak i validaciona tačnost po epohama")
fig.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(f"sacuvano: {OUT_PATH}")
