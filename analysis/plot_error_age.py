import numpy as np
import matplotlib.pyplot as plt
import os

def plot_error_vs_age(npz_path, save_path=None, title=None):
    data = np.load(npz_path)

    y = data["y"]
    y_hat = data["y_hat"]
    err = np.abs(y_hat - y)

    plt.figure(figsize=(5, 4))
    plt.scatter(y, err, s=8, alpha=0.4)
    plt.xlabel("Chronological Age")
    plt.ylabel("Absolute Error (|ŷ − y|)")

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved figure to {save_path}")
    else:
        plt.show()

def plot_error_by_age_bins(npz_path, bins, save_path=None, title=None):
    data = np.load(npz_path)

    y = data["y"]
    y_hat = data["y_hat"]
    err = np.abs(y_hat - y)

    bin_ids = np.digitize(y, bins)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    mae_per_bin = []

    for i in range(1, len(bins)):
        mask = bin_ids == i
        if np.sum(mask) > 0:
            mae_per_bin.append(np.mean(err[mask]))
        else:
            mae_per_bin.append(np.nan)

    plt.figure(figsize=(5, 4))
    plt.plot(bin_centers, mae_per_bin, marker="o")
    plt.xlabel("Age")
    plt.ylabel("MAE")

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # ===== 修改这里 =====
    npz_file = "repr_internal.npz"
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    # 连续散点图
    plot_error_vs_age(
        npz_file,
        save_path=os.path.join(out_dir, "error_vs_age.png"),
        title="Error vs Age"
    )

    # 年龄分箱 MAE（你可以改 bins）
    bins = np.arange(20, 90, 10)
    plot_error_by_age_bins(
        npz_file,
        bins=bins,
        save_path=os.path.join(out_dir, "mae_by_age_bins.png"),
        title="MAE by Age Group"
    )
