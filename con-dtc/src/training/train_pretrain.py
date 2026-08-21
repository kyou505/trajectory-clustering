import argparse
from pathlib import Path

import torch

from src.data.data_loader import create_data_loaders
from src.data.data_process import QDTrajectoryDataset
from src.models.pretrain_model import (
    MSTMLoss,
    STTraj2VecPretrainModel
)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_batch_to_device(batch, device):
    result = {}

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v

    return result

def masked_accuracy(logits, targets):
    # logits: [B, L. V]
    # targets: [B, L]
    valid_mask = targets.ne(-100)
    predictions = logits.argmax(dim=-1)
    correct = (predictions[valid_mask] == targets[valid_mask]).sum()
    total = valid_mask.sum()
    if total > 0:
        return correct.item() / total.item()
    else:
        return 0

def masked_correct_and_total(logits, targets):
    valid_mask = targets.ne(-100)
    predictions = logits.argmax(dim=-1)
    correct = (predictions[valid_mask] == targets[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    return correct, total


def overfit_one_batch(
        num_steps=30,
        batch_size=8
):
    torch.manual_seed(42)
    device = get_device()
    print("device: ", device)
    train_loader, _, _ = create_data_loaders(batch_size=batch_size)
    batch = next(iter(train_loader))
    batch = move_batch_to_device(batch, device)
    model = STTraj2VecPretrainModel().to(device)
    criterion = MSTMLoss(time_loss_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
    initial_loss = None
    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()
        output = model(
            location_ids=batch["masked_location_ids"],
            time_ids=batch["masked_time_ids"],
            attention_mask=batch["attention_mask"],
        )
        losses = criterion(
            location_logits=output["location_logits"],
            time_logits=output["time_logits"],
            location_targets=batch["location_targets"],
            time_targets=batch["time_targets"],
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        location_accuracy = masked_accuracy(output["location_logits"], batch["location_targets"])
        time_accuracy = masked_accuracy(output["time_logits"], batch["time_targets"])

        current_loss = losses["loss"].item()
        if initial_loss is None:
            initial_loss = current_loss

        print(
            f" step={1 + step:3d}"
            f" loss={current_loss:.3f}"
            f" location_loss={losses['location_loss']:.3f}"
            f" time_loss={losses['time_loss']:.3f}"
            f" location_accuracy={location_accuracy:.3f}"
            f" time_accuracy={time_accuracy:.3f}"
        )

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
        max_batches=None,
        log_interval=50,
):
    model.train()
    location_loss_sum = 0.0
    time_loss_sum = 0.0
    location_correct = 0
    location_total = 0
    time_correct = 0
    time_total = 0
    processed_batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            location_ids=batch["masked_location_ids"],
            time_ids=batch["masked_time_ids"],
            attention_mask=batch["attention_mask"],
        )
        losses = criterion(
            location_logits=output["location_logits"],
            time_logits=output["time_logits"],
            location_targets=batch["location_targets"],
            time_targets=batch["time_targets"],
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 当前batch参与loss的token
        current_location_total = batch["location_targets"].ne(-100).sum().item()
        current_time_total = batch["time_targets"].ne(-100).sum().item()
        location_loss_sum += losses["location_loss"].item() * current_location_total
        time_loss_sum += losses["time_loss"].item() * current_time_total

        current_location_correct, current_location_total = masked_correct_and_total(
            output["location_logits"],
            batch["location_targets"]
        )
        location_correct += current_location_correct
        location_total += current_location_total

        current_time_correct, current_time_total = masked_correct_and_total(
            output["time_logits"],
            batch["time_targets"]
        )
        time_correct += current_time_correct
        time_total += current_time_total

        processed_batches += 1
        if (log_interval is not None and (processed_batches) % log_interval == 0):
            print(
                f" batch={processed_batches:04d}"
                f" loss={losses['loss'].item():.4f}"
            )

    if processed_batches == 0:
        raise RuntimeError("No batches were processed during training")
    average_location_loss = location_loss_sum / location_total
    average_time_loss = time_loss_sum / time_total
    average_total_loss = average_location_loss + average_time_loss * criterion.time_loss_weight
    return {
        "loss": average_total_loss,
        "location_loss": average_location_loss,
        "time_loss": average_time_loss,
        "location_accuracy": location_correct / location_total,
        "time_accuracy": time_correct / time_total,
        "batches": processed_batches,
    }

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    max_batches=None,
):
    model.eval()

    location_loss_sum = 0.0
    time_loss_sum = 0.0

    location_correct = 0
    location_total = 0

    time_correct = 0
    time_total = 0

    processed_batches = 0

    for batch_index, batch in enumerate(loader):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        batch = move_batch_to_device(
            batch,
            device,
        )

        output = model(
            location_ids=batch["masked_location_ids"],
            time_ids=batch["masked_time_ids"],
            attention_mask=batch["attention_mask"],
        )

        losses = criterion(
            location_logits=output["location_logits"],
            time_logits=output["time_logits"],
            location_targets=batch["location_targets"],
            time_targets=batch["time_targets"],
        )

        current_location_total = (
            batch["location_targets"]
            .ne(-100)
            .sum()
            .item()
        )

        current_time_total = (
            batch["time_targets"]
            .ne(-100)
            .sum()
            .item()
        )

        location_loss_sum += (
            losses["location_loss"].item()
            * current_location_total
        )

        time_loss_sum += (
            losses["time_loss"].item()
            * current_time_total
        )

        current_correct, current_total = (
            masked_correct_and_total(
                output["location_logits"],
                batch["location_targets"],
            )
        )

        location_correct += current_correct
        location_total += current_total

        current_correct, current_total = (
            masked_correct_and_total(
                output["time_logits"],
                batch["time_targets"],
            )
        )

        time_correct += current_correct
        time_total += current_total

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No batches were processed during evaluation"
        )

    average_location_loss = (
        location_loss_sum / location_total
    )

    average_time_loss = (
        time_loss_sum / time_total
    )

    average_total_loss = (
        average_location_loss
        + criterion.time_loss_weight
        * average_time_loss
    )

    return {
        "loss": average_total_loss,
        "location_loss": average_location_loss,
        "time_loss": average_time_loss,
        "location_accuracy": (
            location_correct / location_total
        ),
        "time_accuracy": (
            time_correct / time_total
        ),
        "batches": processed_batches,
    }

def pretrain(
        num_epochs=10,
        batch_size=32,
        learning_rate=1.5e-4,
        time_loss_weight=0.1,
        weight_decay=0.01,
        seed=42,
        max_train_batches=None,
        max_valid_batches=None,
        dataset_name="qdTimeNoise0424",
        checkpoint_path=None,
):
    torch.manual_seed(seed)
    device = get_device()
    base_dataset = QDTrajectoryDataset(dataset_name=dataset_name)
    train_loader, valid_loader, _ = create_data_loaders(
        batch_size=batch_size,
        seed=seed,
        base_dataset=base_dataset,
    )
    model = STTraj2VecPretrainModel(
        location_vocab_size=base_dataset.location_vocab_size,
        time_vocab_size=base_dataset.time_vocab_size,
        max_length=base_dataset.max_length,
    ).to(device)
    criterion=MSTMLoss(time_loss_weight=time_loss_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    project_dir = Path(__file__).resolve().parents[2]
    checkpoint_dir = project_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is None:
        checkpoint_path = checkpoint_dir / "sttraj2vec_pretrain_best.pt"
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_dir / checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_valid_loss = float("inf")
    history = []
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch} of {num_epochs}")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )
        print(
            "Train: "
            f"loss={train_metrics['loss']:.4f} "
            f"loc_loss={train_metrics['location_loss']:.4f} "
            f"time_loss={train_metrics['time_loss']:.4f} "
            f"loc_acc={train_metrics['location_accuracy']:.4f} "
            f"time_acc={train_metrics['time_accuracy']:.4f}"
        )

        valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            max_batches=max_valid_batches,
        )
        print(
            "Valid: "
            f"loss={valid_metrics['loss']:.4f} "
            f"loc_loss={valid_metrics['location_loss']:.4f} "
            f"time_loss={valid_metrics['time_loss']:.4f} "
            f"loc_acc={valid_metrics['location_accuracy']:.4f} "
            f"time_acc={valid_metrics['time_accuracy']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        })

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_metrics": valid_metrics,
                    "configs": {
                        "dataset": dataset_name,
                        "location_vocab_size": base_dataset.location_vocab_size,
                        "time_vocab_size": base_dataset.time_vocab_size,
                        "max_length": base_dataset.max_length,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "time_loss_weight": time_loss_weight,
                        "weight_decay": weight_decay,
                        "seed": seed,
                    }
                },
                checkpoint_path,
            )

            print("saved best checkpoint:", checkpoint_path)

    print("Best valid loss:", best_valid_loss)
    return model, history

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain STTraj2Vec")
    parser.add_argument("--dataset", default="qdTimeNoise0424")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--time-loss-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    parser.add_argument("--checkpoint-path")
    return parser.parse_args()


def main():
    args = parse_args()
    pretrain(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        time_loss_weight=args.time_loss_weight,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_train_batches=args.max_train_batches,
        max_valid_batches=args.max_valid_batches,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint_path,
    )

if __name__ == "__main__":
    main()
