import torch
from torch import nn

from src.models.dec import target_distribution
from src.models.dec import (ConDTCClusteringLoss)
from src.models.pretrain_model import (MSTMLoss)

class ConDTCTotalLoss(nn.Module):
    def __init__(
            self,
            time_loss_weight=0.1,
            clustering_loss_weight=0.5
    ):
        super().__init__()
        self.clustering_loss_weight = clustering_loss_weight
        self.mstm_loss = MSTMLoss(
            time_loss_weight=time_loss_weight,
        )
        self.clustering_loss = ConDTCClusteringLoss()

    def forward(
            self,
            location_logits,
            time_logits,
            location_targets,
            time_targets,
            q1,
            q2,
            p
    ):
        representation_losses = self.mstm_loss(
            location_logits=location_logits,
            time_logits=time_logits,
            location_targets=location_targets,
            time_targets=time_targets,
        )
        clustering_losses = self.clustering_loss(
            q1=q1,
            q2=q2,
            p=p,
        )
        total_loss = representation_losses["loss"] + clustering_losses["loss"] * self.clustering_loss_weight
        return {
            "loss": total_loss,
            "representation_loss": representation_losses["loss"],
            "location_loss": representation_losses["location_loss"],
            "time_loss": representation_losses["time_loss"],
            "clustering_loss": clustering_losses["loss"],
            "clustering_loss_view1": clustering_losses["loss_view1"],
            "clustering_loss_view2": clustering_losses["loss_view2"],
        }

def test():
    from pathlib import Path
    from src.data_loader import (
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
        clustering_loss_weight=0.5,
    )
    losses = criterion(
        location_logits=mstm_output["location_logits"],
        time_logits=mstm_output["time_logits"],
        location_targets=batch["location_targets"].to(device),
        time_targets=batch["time_targets"].to(device),
        q1=cluster_output["q1"],
        q2=cluster_output["q2"],
        p=p,
    )
    for name, value in losses.items():
        print(name, value.item())
        assert torch.isfinite(value)

if __name__ == "__main__":
    test()