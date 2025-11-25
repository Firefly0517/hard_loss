import torch
import torch.nn as nn
import torch.nn.functional as F

class HardnessAwareKernelizedSupCon(nn.Module):
    """
    Hardness-aware Supervised Contrastive Loss
    (extension of KernelizedSupCon with difficulty reweighting)

    Features:
      - Keeps kernel similarity weighting w_{i,k} = K(y_i - y_k)
      - Adds hardness weights based on inconsistency between label and embedding distances
      - Supports 'hardaware' mode
    """

    def __init__(self, temperature=0.07, base_temperature=0.07,
                 kernel=None, lambda_hard=1.0, hmax=1.0, eps=1e-8,
                 distance_metric='cosine'):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.kernel = kernel
        self.lambda_hard = lambda_hard
        self.hmax = hmax
        self.eps = eps
        self.distance_metric = distance_metric

    def __repr__(self):
        return f"{self.__class__.__name__}(temp={self.temperature}, λ={self.lambda_hard}, hmax={self.hmax})"

    def pairwise_distance(self, z):
        """Compute pairwise distance matrix in embedding space."""
        if self.distance_metric == 'cosine':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
            dist = 1 - sim
        elif self.distance_metric == 'euclidean':
            dist = torch.cdist(z, z, p=2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        return dist

    def forward(self, features, labels):
        """
        Args:
            features: [bsz, n_views, dim] -> 模型输出的特征张量，维度为[批次大小, 视图数, 特征维度]
                  （n_views：数据增强生成的多视图，如同一图像的不同裁剪/旋转）
            labels: [bsz, 1] -> 连续标签张量，维度为[批次大小, 1]
        """
        device = features.device

        if len(features.shape) != 3: # 输入维度检测校验
            raise ValueError('`features` needs to be [bsz, n_views, n_feats],'
                             '3 dimensions are required')

        bsz, n_views, dim = features.shape
        features = F.normalize(features, dim=-1) # L2归一化

        # merge views
        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, dim]
        labels = labels.view(-1, 1)

        if labels.shape[0] != bsz:
            raise ValueError('Num of labels does not match num of features')

        labels = labels.repeat(n_views, 1)
        N = labels.size(0)

        # compute label kernel weights
        w = self.kernel(labels)  # [N, N], e.g. RBF kernel
        w = w.clamp(0., 1.)

        # ----- compute distances -----
        d_y = torch.abs(labels - labels.T)  # label-space distance
        d_y = d_y / (d_y.max() + self.eps)  # normalize

        d_z = self.pairwise_distance(features)  # representation-space distance
        d_z = d_z / (d_z.max() + self.eps)      # normalize

        # inconsistency u_{i,k} = d_z - d_y
        u = d_z - d_y
        u = torch.clamp(u, -self.hmax, self.hmax)

        # compute difficulty multipliers
        m = 1.0 + self.lambda_hard * F.relu(u)   # hard positive
        n = 1.0 + self.lambda_hard * F.relu(-u)  # hard negative

        # adjusted weight
        w_hat = w * m + (1 - w) * n

        # similarity matrix
        sim = torch.div(torch.matmul(features, features.T), self.temperature)

        # mask out self-contrast
        logits_mask = torch.ones_like(sim) - torch.eye(N, device=device)
        exp_sim = torch.exp(sim) * logits_mask

        # compute log-prob
        pos = exp_sim * w_hat  # positive contribution
        neg = exp_sim * (1 - w_hat)  # negative contribution
        denom = (pos + neg).sum(1, keepdim=True) + self.eps

        # numerator also weighted because positives are weighted
        positive_sim = sim * w_hat

        log_prob = positive_sim - torch.log(denom)

        # positive mask (weighted)
        pos_mask = w_hat * logits_mask

        # final contrastive regression loss
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + self.eps)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# ==== Example ====
if __name__ == '__main__':
    # example RBF kernel on labels
    def rbf_kernel(y, sigma=5.0):
        d = torch.abs(y - y.T)
        return torch.exp(-d ** 2 / (2 * sigma ** 2))

    model = HardnessAwareKernelizedSupCon(kernel=lambda y: rbf_kernel(y, sigma=10.0),
                                          lambda_hard=2.0,
                                          hmax=0.5)

    feats = torch.randn(32, 2, 64)
    labels = torch.linspace(20, 80, 32).unsqueeze(1)
    loss = model(feats, labels)
    print('Loss:', loss.item())
