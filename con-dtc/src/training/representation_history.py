"""固定原始输入的逐轮诊断；保存在线模型输出，不参与损失或目标构造。"""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler


@contextmanager
def preserve_training_state(model):
    # 即使shuffle=False，迭代DataLoader也可能消耗torch随机数。
    modes = [(module, module.training) for module in model.modules()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


class RepresentationHistory:
    def __init__(self, output_dir, dataset_name):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.previous = None
        self.summary = []

    def save(self, model, loader, device, *, epoch, phase, projector=None):
        """始终评估全量原始数据；不受训练冒烟测试的max_batches限制。"""
        if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
            raise ValueError("representation history requires a full sequential loader")
        if loader.num_workers != 0 or loader.generator is not None:
            raise ValueError("representation history requires num_workers=0 and default generator")
        expected_epoch = 0 if self.previous is None else self.previous["epoch"] + 1
        if epoch != expected_epoch:
            raise ValueError(f"expected consecutive epoch {expected_epoch}, got {epoch}")
        path = self.output_dir / f"epoch_{epoch:03d}.pt"
        if path.exists():
            raise FileExistsError(f"representation snapshot already exists: {path}")
        keys = ("location_ids", "time_ids", "attention_mask", "pooling_mask")
        # 每个字段独立累计，保持相同样本顺序时，哈希不依赖推理批大小。
        hashes = {key: hashlib.sha256() for key in keys}
        parts = {key: [] for key in ("sample_ids", "labels", "lengths", "embeddings", "q")}
        if projector is not None:
            parts["projected_embeddings"] = []
        if self.previous is not None and (projector is not None) != ("projected_embeddings" in self.previous):
            raise ValueError("projection diagnostics must remain enabled or disabled across epochs")
        with ExitStack() as stack:
            stack.enter_context(preserve_training_state(model))
            if projector is not None:
                stack.enter_context(preserve_training_state(projector))
            for batch in loader:
                for key in keys:
                    value = batch[key].detach().cpu().contiguous()
                    hashes[key].update(value.numpy().tobytes())
                view = {key: batch[key].to(device) for key in keys}
                z = model.encode(view)
                q = model.clustering_layer(z)
                if projector is not None:
                    parts["projected_embeddings"].append(projector(z).detach().cpu().clone())
                for key, value in (("sample_ids", batch["index"]), ("labels", batch["label"]),
                                   ("lengths", batch["length"]), ("embeddings", z), ("q", q)):
                    parts[key].append(value.detach().cpu().clone())
            if not parts["embeddings"]:
                raise ValueError("cannot save representation history for an empty dataset")
            snapshot = {key: torch.cat(values) for key, values in parts.items()}
            snapshot["cluster_centers"] = model.clustering_layer.cluster_centers.detach().cpu().clone()
        ids = snapshot["sample_ids"]
        if not torch.equal(ids, torch.arange(len(loader.dataset))):
            raise ValueError("representation history requires unique IDs 0..N-1 in dataset order")
        for key in ("embeddings", "q", "cluster_centers"):
            if not torch.isfinite(snapshot[key]).all():
                raise ValueError(f"non-finite {key} in representation history")
        if not torch.allclose(snapshot["q"].sum(1), torch.ones(len(ids)), atol=1e-5):
            raise ValueError("representation q rows must sum to one")
        field_hashes = {key: value.hexdigest() for key, value in hashes.items()}
        input_hash = hashlib.sha256(json.dumps(field_hashes, sort_keys=True).encode()).hexdigest()
        snapshot.update({
            "schema_version": 3 if projector is not None else 2, "epoch": epoch, "phase": phase,
            "dataset": self.dataset_name, "input_kind": "original_unaugmented",
            "input_sha256": input_hash, "input_field_sha256": field_hashes,
            "predicted_clusters": snapshot["q"].argmax(1),
            "previous_epoch": None if self.previous is None else self.previous["epoch"],
        })
        norms = snapshot["embeddings"].norm(dim=1)
        snapshot["embedding_norms"] = norms
        row = {"epoch": epoch, "phase": phase, "samples": len(ids), "input_sha256": input_hash,
               "embedding_norm_mean": float(norms.mean()), "zero_norm_n": int((norms <= 1e-12).sum())}
        if projector is not None:
            projected = snapshot["projected_embeddings"]
            if not torch.isfinite(projected).all():
                raise ValueError("non-finite projected embeddings in representation history")
            projection_norms = projected.norm(dim=1)
            snapshot["projection_norms"] = projection_norms
            snapshot["projection_kind"] = "lfss_online_projector_eval"
            row.update(projection_norm_mean=float(projection_norms.mean()),
                       projection_zero_norm_n=int((projection_norms <= 1e-12).sum()))
        if self.previous is not None:
            old = self.previous
            for key in ("sample_ids", "labels", "lengths"):
                if not torch.equal(snapshot[key], old[key]):
                    raise ValueError(f"{key} changed between representation snapshots")
            if input_hash != old["input_sha256"]:
                raise ValueError("original diagnostic inputs changed between epochs")
            valid = (norms > 1e-12) & (old["embedding_norms"] > 1e-12)
            cosine = F.cosine_similarity(snapshot["embeddings"], old["embeddings"], dim=1, eps=1e-12)
            cosine[~valid] = float("nan")
            snapshot["representation_cosine_previous"] = cosine
            snapshot["representation_cosine_valid"] = valid
            if valid.any():
                values = cosine[valid]
                row.update({"cosine_mean": float(values.mean()), "cosine_p10": float(torch.quantile(values, 0.1)),
                            "cosine_median": float(values.median()), "cosine_valid_n": int(valid.sum())})
            else:
                row.update({"cosine_mean": None, "cosine_p10": None, "cosine_median": None, "cosine_valid_n": 0})
            if projector is not None:
                valid_h = (projection_norms > 1e-12) & (old["projection_norms"] > 1e-12)
                cosine_h = F.cosine_similarity(projected, old["projected_embeddings"], dim=1, eps=1e-12)
                cosine_h[~valid_h] = float("nan")
                snapshot["projection_cosine_previous"] = cosine_h
                snapshot["projection_cosine_valid"] = valid_h
                values_h = cosine_h[valid_h]
                row.update(projection_cosine_valid_n=int(valid_h.sum()),
                           projection_cosine_mean=float(values_h.mean()) if valid_h.any() else None,
                           projection_cosine_p10=float(torch.quantile(values_h, 0.1)) if valid_h.any() else None,
                           projection_cosine_median=float(values_h.median()) if valid_h.any() else None)
        # 元数据只用于离线按轨迹类别分组，不进入训练图。
        data = getattr(loader.dataset, "data", None)
        if data is not None and "temporal_mode" in data:
            snapshot["temporal_modes"] = data["temporal_mode"].astype(str).tolist()
        temporary = path.with_suffix(".pt.tmp")
        torch.save(snapshot, temporary)
        temporary.replace(path)
        self.previous = snapshot
        self.summary.append(row)
        summary_path = self.output_dir / "summary.json"
        temporary_summary = summary_path.with_suffix(".json.tmp")
        temporary_summary.write_text(json.dumps(self.summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary_summary.replace(summary_path)
        print(f"representation diagnostic saved: {path} samples={len(ids)} "
              f"cosine_previous={row.get('cosine_mean', 'n/a')}")
        return path
