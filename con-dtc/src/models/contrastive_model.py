from pathlib import Path

import torch
from torch import nn

from src.models.dec import (
    DECClusteringLayer,
    ConDTCClusteringLoss
)
from src.models.encoder import (
    TrajectoryEncoder,
    mask_mean_pooling,
)


class ContrastiveTrajectoryModel(nn.Module):
    def __init__(
            self,
            location_vocab_size=148,
            time_vocab_size=1444,
            d_model=256,
            max_length=62,
            num_heads=2,
            num_layers=2,
            dim_feedforward=1024,
            dropout=0.1,
            num_clusters=12,
    ):
        super().__init__()
        self.encoder = TrajectoryEncoder(
            location_vocab_size=location_vocab_size,
            time_vocab_size=time_vocab_size,
            d_model=d_model,
            max_length=max_length,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.clustering_layer = DECClusteringLayer(
            num_clusters=num_clusters,
            embedding_dim=d_model,
        )
        self.prediction_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        self.location_head = nn.Linear(d_model, location_vocab_size, bias=False)
        self.time_head = nn.Linear(d_model, time_vocab_size, bias=False)
        self.location_head.weight = self.encoder.embedding.location_embedding.weight
        self.time_head.weight = self.encoder.embedding.time_embedding.weight

    def encode(self, view):
        hidden_states = self.encoder(
            location_ids=view["location_ids"],
            time_ids=view["time_ids"],
            attention_mask=view["attention_mask"],
        )
        trajectory_vector = mask_mean_pooling(
            hidden_states=hidden_states,
            pooling_mask=view["pooling_mask"],
        )
        return trajectory_vector

    def forward(self, view1, view2):
        z1 = self.encode(view1)
        z2 = self.encode(view2)
        q1 = self.clustering_layer(z1)
        q2 = self.clustering_layer(z2)
        return {
            "z1": z1,
            "z2": z2,
            "q1": q1,
            "q2": q2,
        }

    def forward_mstm(
            self,
            masked_location_ids,
            masked_time_ids,
            attention_mask,
    ):
        hidden_states = self.encoder(
            location_ids=masked_location_ids,
            time_ids=masked_time_ids,
            attention_mask=attention_mask,
        )
        prediction_hidden = self.prediction_transform(hidden_states)
        location_logits = self.location_head(prediction_hidden)
        time_logits = self.time_head(prediction_hidden)
        return {
            "hidden_states": hidden_states,
            "location_logits": location_logits,
            "time_logits": time_logits,
        }

    def load_pretrained_components(
            self,
            checkpoint_path,
            map_location="cpu",
    ):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        if "model_state_dict" not in checkpoint:
            raise KeyError("checkpoint_path does not contain 'model_state_dict'")
        pretrained_state = checkpoint["model_state_dict"]

        incompatible = self.load_state_dict(pretrained_state, strict=False)
        missing_keys = set(incompatible.missing_keys)
        unexpected_keys = set(incompatible.unexpected_keys)

        expected_missing = {"clustering_layer.cluster_centers"}
        if missing_keys != expected_missing:
            raise RuntimeError(f"unexpected missing keys: {missing_keys}")

        return {
            "epoch": checkpoint.get("epoch"),
            "valid_metrics": checkpoint.get("valid_metrics"),
            "config": checkpoint.get("config"),
        }

def test():
    from src.data_loader import (
        create_contrastive_data_loader,
    )

    loader = create_contrastive_data_loader(
        batch_size=8,
        seed=42,
        shuffle=False,
    )
    batch = next(iter(loader))
    # model = ContrastiveTrajectoryModel()
    # output = model(view1=batch["view1"], view2=batch["view2"])
    # print(
    #     "z1: {}, z2: {}".format(output["z1"].shape, output["z2"].shape)
    # )
    # print(
    #     "q1: {}, q2: {}".format(output["q1"].shape, output["q2"].shape)
    # )
    # criterion = ConDTCClusteringLoss()
    # losses = criterion(output["q1"], output["q2"])
    # print("contrastive loss:", losses["loss"].item())
    # losses["loss"].backward()
    # assert model.clustering_layer.cluster_centers.grad is not None

    project_dir = Path(__file__).resolve().parents[2]
    checkpoint_path = project_dir / "checkpoints" / "sttraj2vec_pretrain_best.pt"
    model = ContrastiveTrajectoryModel()
    metadata = model.load_pretrained_components(checkpoint_path, map_location="cpu")
    print("loaded checkpoint:", checkpoint_path)
    print("pretrain epoch:", metadata["epoch"])
    print("valid metrics:", metadata["valid_metrics"])
    mstm_output = model.forward_mstm(
        masked_location_ids=batch["masked_location_ids"],
        masked_time_ids=batch["masked_time_ids"],
        attention_mask=batch["attention_mask"],
    )
    print("location logits:",mstm_output["location_logits"].shape)
    print("time logits:",mstm_output["time_logits"].shape)


if __name__ == "__main__":
    test()