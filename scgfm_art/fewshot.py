from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def class_cap_indices(
    labels: torch.Tensor,
    samples_per_class: int,
    seed: int,
) -> list[int]:
    """Match SCGFM-ART's deterministic per-class center sampling."""
    if samples_per_class < 0:
        raise ValueError("samples_per_class must be non-negative.")
    if samples_per_class == 0:
        return list(range(labels.numel()))
    rng = np.random.default_rng(seed)
    labels_numpy = labels.detach().cpu().numpy()
    selected: list[int] = []
    for label in np.unique(labels_numpy):
        candidates = np.flatnonzero(labels_numpy == label)
        if candidates.size > samples_per_class:
            candidates = rng.choice(
                candidates, samples_per_class, replace=False
            )
        selected.extend(int(index) for index in candidates)
    rng.shuffle(selected)
    return selected


def create_split(
    labels: np.ndarray,
    k_shot: int,
    n_query: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    support, query = [], []
    for class_id in np.unique(labels):
        indices = np.where(labels == class_id)[0]
        rng.shuffle(indices)
        if len(indices) < k_shot:
            cut = max(1, len(indices) - 1)
        else:
            cut = k_shot
        support.extend(indices[:cut].tolist())
        available = indices[cut:]
        query.extend(
            (
                available
                if n_query == -1
                else available[:n_query]
            ).tolist()
        )
    return support, query


def support_standardize(
    support: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit normalization on support only; never use query statistics."""
    mean = support.mean(dim=0, keepdim=True)
    scale = support.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (support - mean) / scale, (query - mean) / scale


def _remap_labels(
    support_labels: torch.Tensor,
    query_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    classes = torch.unique(support_labels, sorted=True)
    support_mapped = torch.searchsorted(classes, support_labels)
    query_mapped = torch.searchsorted(classes, query_labels)
    if torch.any(query_mapped >= classes.numel()) or torch.any(
        classes[query_mapped] != query_labels
    ):
        raise ValueError("Every query class must be present in the support set.")
    return support_mapped, query_mapped, classes


def _prototype_probabilities(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    prototypes = torch.stack(
        [
            support[support_labels == class_id].mean(dim=0)
            for class_id in range(num_classes)
        ]
    )
    return torch.softmax(-torch.cdist(query, prototypes, p=2), dim=1)


def _fit_baseline_linear_head(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int,
    max_iter: int,
    lr: float,
) -> nn.Module:
    """Match the SCGFM-ART linear-probe optimizer and objective."""
    head = LinearProbe(support.shape[1], num_classes).to(support.device)
    nn.init.zeros_(head.classifier.weight)
    nn.init.zeros_(head.classifier.bias)
    counts = torch.bincount(
        support_labels, minlength=num_classes
    ).float()
    class_weights = support_labels.numel() / (
        num_classes * counts.clamp_min(1.0)
    )
    optimizer = torch.optim.LBFGS(
        head.parameters(),
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=1e-5,
        tolerance_change=1e-7,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    head.train()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            head(support),
            support_labels,
            weight=class_weights,
        )
        loss = (
            loss
            + 0.5
            * head.classifier.weight.square().sum()
            / support_labels.numel()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return head.eval()


def _record_metrics(
    run: int,
    head_name: str,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
) -> dict:
    prediction = probabilities.argmax(dim=1)
    return {
        "run": run,
        "head": head_name,
        "accuracy": float((prediction == labels).float().mean()),
        "query_size": int(labels.numel()),
    }


def _summarize(rows: list[dict], head_name: str) -> dict:
    selected = [row for row in rows if row["head"] == head_name]
    summary = {"n_runs": len(selected)}
    values = [float(row["accuracy"]) for row in selected]
    if values:
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std(ddof=0))
        ci95 = float(1.96 * std / math.sqrt(len(array)))
    else:
        mean, std, ci95 = 0.0, 0.0, 0.0
    summary["accuracy_mean"] = mean
    summary["accuracy_std"] = std
    summary["accuracy_ci95"] = ci95
    return summary


def evaluate_fewshot(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_shot: int,
    n_query: int,
    n_runs: int,
    seed: int,
    device: str | torch.device,
) -> tuple[dict, list[dict]]:
    """Evaluate the paper's graph-level 5-shot ProtoNet protocol."""
    target_device = torch.device(device)
    features = embeddings.detach().float().cpu()
    target = labels.detach().long().cpu()
    target_numpy = target.numpy()
    rows: list[dict] = []

    for run in range(n_runs):
        run_seed = seed + run
        torch.manual_seed(run_seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)
        support_ids, query_ids = create_split(
            target_numpy, k_shot, n_query, run_seed
        )
        if not query_ids:
            continue
        support = features[support_ids].to(target_device, non_blocking=True)
        query = features[query_ids].to(target_device, non_blocking=True)
        support, query = support_standardize(support, query)
        support_labels = target[support_ids].to(target_device)
        query_labels = target[query_ids].to(target_device)
        support_labels, query_labels, classes = _remap_labels(
            support_labels, query_labels
        )
        with torch.no_grad():
            probabilities = _prototype_probabilities(
                support,
                support_labels,
                query,
                int(classes.numel()),
            )
        rows.append(
            _record_metrics(
                run + 1, "prototype", query_labels, probabilities
            )
        )

    return {
        "k_shot": k_shot,
        "n_query": n_query,
        "requested_runs": n_runs,
        "heads": {"prototype": _summarize(rows, "prototype")},
    }, rows

def create_episode_splits(
    labels: torch.Tensor,
    *,
    k_shot: int,
    n_query: int,
    n_runs: int,
    seed: int,
) -> list[dict[str, torch.Tensor | int]]:
    """Create deterministic support/query splits from the complete node pool."""
    target = labels.detach().long().cpu()
    target_numpy = target.numpy()
    splits: list[dict[str, torch.Tensor | int]] = []
    for run in range(n_runs):
        support, query = create_split(
            target_numpy, k_shot, n_query, seed + run
        )
        if not query:
            continue
        splits.append(
            {
                "run": run + 1,
                "seed": seed + run,
                "support_indices": torch.tensor(
                    support, dtype=torch.long
                ),
                "query_indices": torch.tensor(query, dtype=torch.long),
            }
        )
    return splits


def episode_index_union(
    splits: list[dict[str, torch.Tensor | int]],
) -> torch.Tensor:
    values = []
    for split in splits:
        values.extend(
            [
                torch.as_tensor(split["support_indices"]).long(),
                torch.as_tensor(split["query_indices"]).long(),
            ]
        )
    if not values:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.cat(values), sorted=True)


def evaluate_fewshot_splits(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    index_to_embedding_row: torch.Tensor,
    splits: list[dict[str, torch.Tensor | int]],
    *,
    device: str | torch.device,
    linear_epochs: int = 1000,
    linear_lr: float = 1.0,
    heads: tuple[str, ...] | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate fixed episodes with the released linear-head protocol."""
    if heads is None:
        heads = ("linear",)
    allowed_heads = {"linear"}
    unknown_heads = set(heads) - allowed_heads
    if unknown_heads:
        raise ValueError(
            f"Unknown few-shot heads: {sorted(unknown_heads)}; "
            f"expected a subset of {sorted(allowed_heads)}."
        )
    if not heads:
        raise ValueError("At least one few-shot head must be requested.")
    target_device = torch.device(device)
    features = embeddings.detach().float().cpu()
    target = labels.detach().long().cpu()
    row_map = index_to_embedding_row.detach().long().cpu()
    rows: list[dict] = []

    for split in splits:
        run = int(split["run"])
        run_seed = int(split["seed"])
        torch.manual_seed(run_seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)
        support_ids = torch.as_tensor(split["support_indices"]).long()
        query_ids = torch.as_tensor(split["query_indices"]).long()
        support_rows = row_map[support_ids]
        query_rows = row_map[query_ids]
        if (support_rows < 0).any() or (query_rows < 0).any():
            raise ValueError("Episode index is missing from encoded union.")
        support = features[support_rows].to(
            target_device, non_blocking=True
        )
        query = features[query_rows].to(
            target_device, non_blocking=True
        )
        support, query = support_standardize(support, query)
        support_labels = target[support_ids].to(target_device)
        query_labels = target[query_ids].to(target_device)
        support_labels, query_labels, classes = _remap_labels(
            support_labels, query_labels
        )
        num_classes = int(classes.numel())

        def record(head: str, probabilities: torch.Tensor) -> None:
            metrics = _record_metrics(
                run, head, query_labels, probabilities
            )
            metrics["seed"] = run_seed
            rows.append(metrics)

        if "linear" in heads:
            linear = _fit_baseline_linear_head(
                support,
                support_labels,
                num_classes=num_classes,
                max_iter=linear_epochs,
                lr=linear_lr,
            )
            with torch.no_grad():
                record("linear", torch.softmax(linear(query), dim=1))

    return {
        "requested_runs": len(splits),
        "protocol": "scgfm_art_linear_v1",
        "heads": {
            name: _summarize(rows, name) for name in heads
        },
    }, rows





