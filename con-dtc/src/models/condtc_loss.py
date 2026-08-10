import torch
from torch import nn
import torch.nn.functional as F

from src.models.dec import target_distribution
from src.models.dec import (ConDTCClusteringLoss)
from src.models.pretrain_model import (MSTMLoss)

class ConDTCTotalLoss(nn.Module):
    def __init__(
            self,
            time_loss_weight=0.1,
            clustering_loss_weight=2,
            instance_temperature=0.5,
            cluster_temperature=1.0,
            instance_loss_weight=1.0,
            cluster_contrastive_loss_weight=1.0,
    ):
        super().__init__()
        self.clustering_loss_weight = clustering_loss_weight
        self.instance_loss_weight = instance_loss_weight
        self.cluster_contrastive_loss_weight = cluster_contrastive_loss_weight
        self.mstm_loss = MSTMLoss(
            time_loss_weight=time_loss_weight,
        )
        self.clustering_loss = ConDTCClusteringLoss()
        self.instance_contrastive_loss = InfoNCELoss(
            temperature=instance_temperature,
        )
        self.cluster_contrastive_loss = InfoNCELoss(
            temperature=cluster_temperature,
        )

    def forward(
            self,
            location_logits,
            time_logits,
            location_targets,
            time_targets,
            q1,
            q2,
            p,
            head_in1,
            head_in2,
            head_cl1,
            head_cl2,
    ):
        representation_losses = self.mstm_loss(
            location_logits=location_logits,
            time_logits=time_logits,
            location_targets=location_targets,
            time_targets=time_targets,
        )
        instance_contrastive_loss = self.instance_contrastive_loss(head_in1, head_in2)
        cluster_contrastive_loss = self.cluster_contrastive_loss(head_cl1, head_cl2)
        clustering_losses = self.clustering_loss(
            q1=q1,
            q2=q2,
            p=p,
        )
        total_loss = (representation_losses["loss"]
                      + clustering_losses["loss"] * self.clustering_loss_weight
                      + cluster_contrastive_loss * self.cluster_contrastive_loss_weight
                      + instance_contrastive_loss * self.instance_loss_weight)
        return {
            "loss": total_loss,
            "representation_loss": representation_losses["loss"],
            "location_loss": representation_losses["location_loss"],
            "time_loss": representation_losses["time_loss"],
            "instance_contrastive_loss": instance_contrastive_loss,
            "cluster_contrastive_loss": cluster_contrastive_loss,
            "clustering_loss": clustering_losses["loss"],
            "clustering_loss_view1": clustering_losses["loss_view1"],
            "clustering_loss_view2": clustering_losses["loss_view2"],
        }

class InfoNCELoss(nn.Module):
    def __init__(
            self,
            temperature
    ):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("Temperature should be positive")
        self.temperature = temperature

    def forward(
            self,
            features1,
            features2,
    ):
        if features1.shape != features2.shape:
            raise ValueError("features1 and features2 should have the same shape")
        if features1.ndim != 2:
            raise ValueError("features1 should have 2 dimensions")

        num_samples = features1.size(0)
        if num_samples < 2:
            raise ValueError("feature1 should have at least 2 features")
        # [N, D] + [N, D] -> [2N, D]
        features = torch.cat((features1, features2), dim=0)
        features = F.normalize(features, p=2, dim=1)
        similarity = (features @ features.transpose(0, 1)) / self.temperature
        # 正样本：(view1[i], view2[i])
        positive_logits = torch.cat([
            similarity.diagonal(offset=num_samples),
            similarity.diagonal(offset=-num_samples),
        ]).unsqueeze(-1)
        # 排除自身以及对应的正样本
        identity = torch.eye(num_samples, dtype=torch.bool, device=features.device)
        negative_mask = ~identity.repeat(2, 2)
        negative_logits = similarity.masked_select(
            negative_mask,
        ).view(2 * num_samples, -1)
        logits = torch.cat(
            [positive_logits, negative_logits],
            dim=1
        )
        labels = torch.zeros(
            2 * num_samples,
            dtype=torch.long,
            device=features.device,
        )
        return F.cross_entropy(logits, labels)


def test():
    from pathlib import Path
    from data.data_loader import (
        create_contrastive_data_loader,
    )
    from src.models.contrastive_model import (
        ContrastiveTrajectoryModel,
    )
    torch.manual_seed(42)
    device = torch.device("cpu")
    loader = create_contrastive_data_loader(
        batch_size=8,
        seed=42,
        shuffle=False,
    )
    batch = next(iter(loader))
    project_dir = Path(__file__).resolve().parent.parent
    checkpoint_path = project_dir / "checkpoints" / "sttraj2vec_pretrain_best.pt"
    model = ContrastiveTrajectoryModel(num_clusters=12).to(device)
    model.load_pretrained_components(
        checkpoint_path,
        map_location=device,
    )
    mstm_output = model.forward_mstm(
        masked_location_ids=batch["masked_location_ids"].to(device),
        masked_time_ids=batch["masked_time_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    view1 = {
        key: value.to(device)
        for key, value in batch["view1"].items()
    }
    view2 = {
        key: value.to(device)
        for key, value in batch["view2"].items()
    }
    cluster_output = model(
        view1=view1,
        view2=view2,
    )
    q_average = 0.5 * (cluster_output["q1"].detach() + cluster_output["q2"].detach())
    p = target_distribution(q_average)
    criterion = ConDTCTotalLoss(
        time_loss_weight=0.1,
        clustering_loss_weight=2,
    )
    losses = criterion(
        location_logits=mstm_output["location_logits"],
        time_logits=mstm_output["time_logits"],
        location_targets=batch["location_targets"].to(device),
        time_targets=batch["time_targets"].to(device),
        q1=cluster_output["q1"],
        q2=cluster_output["q2"],
        p=p,
        head_in1=cluster_output["head_in1"],
        head_in2=cluster_output["head_in2"],
        head_cl1=cluster_output["head_cl1"],
        head_cl2=cluster_output["head_cl2"],
    )
    for name, value in losses.items():
        print(name, value.item())
        assert torch.isfinite(value)

def test_info_nce():
    torch.manual_seed(42)

    features1 = torch.randn(8, 128, requires_grad=True)
    features2 = torch.randn(8, 128, requires_grad=True)

    criterion = InfoNCELoss(temperature=0.5)
    loss = criterion(features1, features2)

    print("InfoNCE loss:", loss.item())


if __name__ == "__main__":
    test()
    test_info_nce()