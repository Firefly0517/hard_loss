import numpy as np
import matplotlib.pyplot as plt
import umap
import os

def plot_umap(npz_path, color_by="age", save_path=None, title=None):
    data = np.load(npz_path)

    z = data["z"]
    y = data["y"]

    if color_by == "age":
        color = y
        cbar_label = "Age"
        cmap = "viridis"
    elif color_by == "error":
        y_hat = data["y_hat"]
        color = np.abs(y_hat - y)
        cbar_label = "Absolute Error"
        cmap = "hot"
    else:
        raise ValueError("color_by must be 'age' or 'error'")

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42
    )

    emb = reducer.fit_transform(z)

    plt.figure(figsize=(5, 4))
    sc = plt.scatter(
        emb[:, 0], emb[:, 1],
        c=color,
        s=8,
        cmap=cmap,
        alpha=0.8
    )
    plt.colorbar(sc, label=cbar_label)

    plt.xticks([])
    plt.yticks([])

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

    # 按真实年龄上色
    plot_umap(
        npz_file,
        color_by="age",
        save_path=os.path.join(out_dir, "umap_by_age.png"),
        title="UMAP (colored by age)"
    )

    # 按预测误差上色
    plot_umap(
        npz_file,
        color_by="error",
        save_path=os.path.join(out_dir, "umap_by_error.png"),
        title="UMAP (colored by error)"
    )
