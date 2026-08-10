
import torch
from torch import nn
import torch.nn.functional as F

class DECClusteringLayer(nn.Module):
    def __init__(
            self,
            num_clusters,
            embedding_dim,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.cluster_centers = nn.Parameter(
            torch.empty(num_clusters, embedding_dim)
        )
        # 测试用
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z):
        """
        :param z: [B, D]
        :return: q: [B, K]
        """
        # [B, 1, D] - [1, K, D] -> [B, K, D]
        difference = z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)
        # ||zi - cj||²，形状为 [B, K]
        difference_squared = difference.pow(2).sum(dim=-1)
        # Student's t-distribution
        numerator = 1.0 / (1.0 + difference_squared)
        # 归一化
        q = numerator / numerator.sum(dim=1, keepdim=True).clamp(min=1e-12)

        return q

@torch.no_grad()
def target_distribution(q):
    cluster_frequency = q.sum(dim=0, keepdim=True).clamp(min=1e-12)
    weight = q.pow(2) / cluster_frequency
    p = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return p

class DECLoss(nn.Module):
    def forward(self, q, p):
        log_q = q.clamp(min=1e-12).log()
        loss = F.kl_div(input=log_q, target=p, reduction='batchmean')
        return loss

class ConDTCClusteringLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_loss = DECLoss()

    def forward(self, q1, q2, p):
        if (q1.shape != q2.shape):
            raise ValueError("q1 and q2 must have the same shape")
        if (q1.shape != p.shape):
            raise ValueError("q1 and p must have the same shape")
        # P是epoch开始时固定计算的目标，不参与梯度
        p = p.detach()
        """
            论文与源码不一致：
        """
        # 论文版本：公式（16）和（18），需要分别计算 P1/P2，并进行跨视图监督
        # p1 = target_distribution(q1)
        # p2 = target_distribution(q2)
        # loss_view1 = self.dec_loss(q=q1, p=p2)
        # loss_view2 = self.dec_loss(q=q2, p=p1)
        # 源码版本 exp2.py L2459
        loss_view1 = self.dec_loss(q1, p)
        loss_view2 = self.dec_loss(q2, p)

        loss = 0.5 * (loss_view1 + loss_view2)
        return {
            "loss": loss,
            "loss_view1": loss_view1,
            "loss_view2": loss_view2
        }


def test():
    torch.manual_seed(42)
    batch_size = 8
    embedding_dim = 128
    num_clusters = 12
    z = torch.randn(batch_size, embedding_dim, requires_grad=True)
    clustering_layer = DECClusteringLayer(num_clusters, embedding_dim)
    q = clustering_layer(z)
    p = target_distribution(q)
    loss = DECLoss()(q, p)
    print("q shape: ", q.shape, "q sum: ", q.sum(dim=1))
    print("p shape: ", p.shape, "p sum: ", p.sum(dim=1))
    print("DEC loss: ", loss.item())

if __name__ == "__main__":
    test()