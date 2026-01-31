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

    def forward(self, features, labels=None):
        """
        Hard-aware y-aware Supervised Contrastive Loss

        Args:
            features: Tensor [bsz, n_views, dim]
            labels:   Tensor [bsz]
        Returns:
            loss: scalar
        """
        device = features.device

        if len(features.shape) != 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, dim]'
            )

        batch_size, n_views, dim = features.shape

        # -------------------------------
        # handle labels / kernel mask
        # -------------------------------
        if labels is None:
            # SimCLR case
            mask = torch.eye(batch_size, device=device)
        else:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            if self.kernel is None:
                # SupCon (discrete labels)
                mask = torch.eq(labels, labels.T).float()
            else:
                # y-aware kernel
                mask = self.kernel(labels).clamp(0., 1.)

        # -------------------------------
        # flatten multi-view features
        # -------------------------------
        view_count = n_views
        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # [N, D]
        N = features.shape[0]

        # repeat labels for views
        if labels is not None:
            labels_all = labels.repeat(view_count, 1)  # [N, 1]
        else:
            labels_all = None

        # repeat mask for multi-view
        mask = mask.repeat(view_count, view_count)  # [N, N]

        # remove self-contrast
        inv_diagonal = torch.ones_like(mask)
        inv_diagonal.fill_diagonal_(0.)

        # -------------------------------
        # similarity logits
        # -------------------------------
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature

        # numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        alignment = logits

        # -------------------------------
        # denominator (uniformity)
        # -------------------------------
        uniformity = torch.exp(logits) * inv_diagonal

        # if self.method == 'threshold':
        #     repeated = mask.unsqueeze(-1).repeat(1, 1, mask.shape[0])
        #     delta = (mask[:, None].T - repeated.T).transpose(1, 2)
        #     delta = (delta > 0.).float()
        #
        #     uniformity = uniformity.unsqueeze(-1).repeat(1, 1, mask.shape[0])
        #     if self.delta_reduction == 'mean':
        #         uniformity = (uniformity * delta).mean(-1)
        #     else:
        #         uniformity = (uniformity * delta).sum(-1)
        #
        # elif self.method == 'expw':
        #     uniformity = torch.exp(logits * (1 - mask)) * inv_diagonal

        uniformity = torch.log(uniformity.sum(1, keepdim=True) + 1e-8)

        # -------------------------------
        # HARD-AWARE PART (new)
        # -------------------------------
        if labels_all is not None and self.lambda_hard > 0:
            # label distance
            d_y = torch.abs(labels_all - labels_all.T)
            d_y = d_y / (d_y.max() + 1e-8)

            # representation distance (cosine)
            sim = torch.matmul(features, features.T)
            d_z = 1.0 - sim
            d_z = d_z / (d_z.max() + 1e-8)

            # inconsistency
            u = torch.clamp(d_z - d_y, -self.hmax, self.hmax)

            # hard-aware modulation
            m = 1.0 + self.lambda_hard * F.relu(u)
        else:
            m = None

        # -------------------------------
        # positive distribution
        # -------------------------------
        if m is not None:
            positive_mask = mask * m * inv_diagonal
        else:
            positive_mask = mask * inv_diagonal

        # -------------------------------
        # log-probability
        # -------------------------------
        log_prob = alignment - uniformity

        pos_sum = positive_mask.sum(1)
        pos_sum = pos_sum.clamp(min=1e-8)

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / pos_sum

        # -------------------------------
        # loss
        # -------------------------------
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
