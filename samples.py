"""
export_openbhb_samples.py

用途：
- 从 DataLoader 或 Dataset 中快速导出若干样本
- 保存为 .npy，用于 debug / 可视化 / 单独跑 loss / 复现实验

适配场景：
- OpenBHB / quasiraw
- 多视角对比学习（images 是 list of views）

作者建议用法：
1. 优先从 DataLoader 导出（与真实训练完全一致）
2. Dataset 导出作为可选补充（无 augmentation）
"""

import os
import numpy as np
import torch


# =============================
# 方式一：从 DataLoader 导出（推荐）
# =============================

def export_from_loader(
    loader,
    save_dir,
    num_batches=1,
    max_samples_per_batch=4
):
    """
    从 DataLoader 中导出样本（支持多视角）

    参数
    ----
    loader : torch.utils.data.DataLoader
        你的 train_loader / val_loader
    save_dir : str
        保存目录
    num_batches : int
        抓取前多少个 batch
    max_samples_per_batch : int
        每个 batch 最多保存多少个样本
    """
    os.makedirs(save_dir, exist_ok=True)

    saved = 0
    for batch_idx, (images, labels, sites) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        # images: list of views
        for v, img_view in enumerate(images):
            # img_view: [B, D, H, W] 或 [B, 1, D, H, W]
            img_view = img_view.cpu().numpy()

            B = img_view.shape[0]
            for i in range(min(B, max_samples_per_batch)):
                sample = img_view[i]
                age = float(labels[i])
                site = sites[i]

                fname = f"sample_{saved:04d}_view{v}_age{age:.1f}.npy"
                np.save(os.path.join(save_dir, fname), sample)
                saved += 1

    print(f"[OK] Saved {saved} samples to {save_dir}")


# =============================
# 方式二：从 Dataset 按 index 导出（无 augmentation）
# =============================

def export_from_dataset(
    dataset,
    indices,
    save_dir
):
    """
    从 Dataset 中按 index 导出样本

    参数
    ----
    dataset : torch.utils.data.Dataset
    indices : list[int]
        要导出的样本 index
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx in indices:
        x, age, site = dataset[idx]

        if torch.is_tensor(x):
            x = x.cpu().numpy()

        fname = f"idx{idx}_age{float(age):.1f}_site{site}.npy"
        np.save(os.path.join(save_dir, fname), x)

    print(f"[OK] Saved {len(indices)} samples to {save_dir}")


# =============================
# 方式三：导出 encoder / projector embedding（进阶）
# =============================

def export_embeddings(
    loader,
    model,
    device,
    save_path,
    max_batches=1
):
    """
    导出模型 embedding（用于 UMAP / hardness 分析）

    保存内容：
    - embeddings: [N, D]
    - ages: [N]
    """
    model.eval()
    feats, ages = [], []

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(loader):
            if i >= max_batches:
                break

            # 拼多视角
            images = torch.cat(images, dim=0).to(device)

            # 确保 channel 维存在
            if images.dim() == 4:
                images = images.unsqueeze(1)

            z = model(images)  # [B * V, D]
            feats.append(z.cpu().numpy())

            # 对齐标签
            V = images.shape[0] // labels.shape[0]
            ages.append(labels.repeat(V).cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    ages = np.concatenate(ages, axis=0)

    np.save(save_path, {
        "embeddings": feats,
        "ages": ages
    })

    print(f"[OK] Saved embeddings to {save_path}")


# =============================
# 使用示例（取消注释即可用）
# =============================

# export_from_loader(
#     train_loader,
#     save_dir="./debug_samples",
#     num_batches=1,
#     max_samples_per_batch=2
# )
#
# export_from_dataset(
#     train_dataset,
#     indices=[0, 10, 42],
#     save_dir="./raw_samples"
# )

# export_embeddings(
#     train_loader,
#     model,
#     device=opts.device,
#     save_path="./debug_embeddings.npy",
#     max_batches=2
# )
