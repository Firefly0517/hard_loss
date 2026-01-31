import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import umap
import yaml

from data import OpenBHB
from main_mse import get_transforms
import models

@torch.no_grad()
def extract_features(model, loader, device='cuda', use_projection=True, max_batches=None):
    model.eval()
    feats, ages, sites = [], [], []

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        images, labels, meta = batch  # 你的 OpenBHB 返回 (images, labels, _)
        # 你 main_infonce.py 训练时 images 是 list(views)，但 test transform 一般是单个 tensor
        # 这里兼容两种情况：
        if isinstance(images, (list, tuple)):
            # 如果是多views，取第一个 view 来做可视化（也可以取平均）
            x = images[0]
        else:
            x = images
        x = x.to(device)

        # labels: 你 cont 标签一般是 age；如果是 [bsz] 就是年龄
        # 有些版本可能是 (age, site) 的结构，这里做兼容
        if labels.ndim == 2 and labels.shape[1] >= 2:
            age = labels[:, 0].cpu().numpy()
            site = labels[:, 1].cpu().numpy()
        else:
            age = labels.cpu().numpy()
            site = None

        # 取特征：projection(128) 或 encoder(512)
        m = model.module if hasattr(model, "module") else model
        if use_projection:
            f = m(x)              # forward -> 128-d normalized
        else:
            f = m.encoder(x)      # encoder -> 512-d

        feats.append(f.cpu().numpy())
        ages.append(age)
        if site is not None:
            sites.append(site)

    feats = np.concatenate(feats, axis=0)
    ages = np.concatenate(ages, axis=0)
    sites = np.concatenate(sites, axis=0) if len(sites) > 0 else None
    return feats, ages, sites

def run_umap(feats, n_neighbors=15, min_dist=0.1, random_state=0):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    emb = reducer.fit_transform(feats)
    return emb

def plot_by_age(emb, ages, out_png):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=ages, s=6)
    plt.colorbar(sc, label="Age")
    plt.title("UMAP colored by Age")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_by_site(emb, sites, out_png):
    plt.figure(figsize=(7, 6))
    # site 是离散值：用不同颜色
    plt.scatter(emb[:, 0], emb[:, 1], c=sites, s=6)
    plt.title("UMAP colored by Site")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="path to yaml config")

    ap.add_argument("--data_dir", type=str)
    ap.add_argument("--ckpt", type=str)
    ap.add_argument("--split", type=str, default="test_external")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpus", type=str, default="0")
    ap.add_argument("--use_projection", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="umap_out")
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)

    args = ap.parse_args()

    # ===== YAML 覆盖 =====
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    # 最终检查
    assert args.data_dir is not None, "data_dir must be specified"
    assert args.ckpt is not None, "ckpt must be specified"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # transforms（用你项目的 get_transforms）
    class O:  # 简单造一个 opts 给 get_transforms 用
        tf = "none"
    T_train, T_test = get_transforms(O())

    # dataset/loader
    if args.split == "train":
        ds = OpenBHB(args.data_dir, train=True, internal=True, transform=T_test, label="cont")
        name = "train"
    elif args.split == "test_internal":
        ds = OpenBHB(args.data_dir, train=False, internal=True, transform=T_test, label="cont")
        name = "internal"
    else:
        ds = OpenBHB(args.data_dir, train=False, internal=False, transform=T_test, label="cont")
        name = "external"

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model
    model = models.SupConResNet("resnet18", feat_dim=128)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # load checkpoint（你的 ckpt 里一般是 {'model': state_dict, ...}）
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # 兼容单卡/多卡 key
    model_state = model.state_dict()
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module.") and k[7:] in model_state:
            new_sd[k[7:]] = v
        elif not k.startswith("module.") and f"module.{k}" in model_state:
            new_sd[f"module.{k}"] = v
        elif k in model_state:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # extract -> umap -> plot
    feats, ages, sites = extract_features(model, loader, device=device, use_projection=bool(args.use_projection))
    emb = run_umap(feats, n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=0)

    tag = "proj128" if args.use_projection else "enc512"
    out_age = os.path.join(args.out_dir, f"umap_{name}_{tag}_age.png")
    plot_by_age(emb, ages, out_age)

    if sites is not None:
        out_site = os.path.join(args.out_dir, f"umap_{name}_{tag}_site.png")
        plot_by_site(emb, sites, out_site)

    print("Saved:", out_age)

if __name__ == "__main__":
    main()
