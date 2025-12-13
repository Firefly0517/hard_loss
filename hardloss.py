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
        device = features.device

        if features.dim() != 3:
            raise ValueError("features must be [bsz, n_views, dim]")

        bsz, n_views, dim = features.shape
        labels = labels.view(bsz, 1)

        # normalize & flatten views
        features = F.normalize(features, dim=-1)
        feats = torch.cat(torch.unbind(features, dim=1), dim=0)  # [N, D]
        labels = labels.repeat(n_views, 1)  # [N, 1]
        N = feats.size(0)

        # mask self-contrast
        logits_mask = torch.ones((N, N), device=device)
        logits_mask.fill_diagonal_(0.)

        # -------- y-aware kernel weight (positives only) --------
        w = self.kernel(labels).clamp(0., 1.)  # [N, N]

        # -------- hardness: inconsistency u = d_z - d_y --------
        d_y = torch.abs(labels - labels.T)
        d_y = d_y / (d_y.max() + self.eps)

        d_z = self.pairwise_distance(feats)
        d_z = d_z / (d_z.max() + self.eps)

        u = torch.clamp(d_z - d_y, -self.hmax, self.hmax)

        # -------- hard-aware positive modulation --------
        m = 1.0 + self.lambda_hard * F.relu(u)

        # final positive weights (THIS is the key)
        # w_pos = (w * m) * logits_mask
        w_pos = w
        # -------- InfoNCE logits (standard) --------
        sim = torch.matmul(feats, feats.T) / self.temperature
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        log_prob = logits - log_denom  # [N, N]

        # -------- y-aware expectation over positives --------
        w_pos_sum = w_pos.sum(dim=1).clamp(min=self.eps)
        mean_log_prob_pos = (w_pos * log_prob).sum(dim=1) / w_pos_sum

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
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
