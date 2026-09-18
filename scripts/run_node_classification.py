from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset

from scgfm_art.checkpoint import load_checkpoint
from scgfm_art.encoder import SCGFMARTEncoder
from scgfm_art.fewshot import (
    class_cap_indices,
    create_episode_splits,
    episode_index_union,
    evaluate_fewshot_splits,
)
from scgfm_art.model import SCGFMARTConfig
from scgfm_art.node_data import (
    EXTERNAL_NODE_TARGETS,
    SOURCE_NODE_DATASETS,
    PreparedNodeSubgraphDataset,
    ShardAwareSampler,
    canonical_node_dataset_name,
)
from scgfm_art.training import pretrain
from scgfm_art.utils import (
    resolve_device,
    set_seed,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_EVALUATION_VERSION = "scgfm_art_node_lodo_v1"
NODE_HEAD_PROTOCOL = "scgfm_art_linear_v1"


def _tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _evaluation_base_protocol(args: argparse.Namespace, model) -> dict:
    mode = getattr(
        args, "evaluation_base_mode", model.config.base_training
    )
    protocol = {
        "mode": mode,
        "normalized_bases_sha256": _tensor_sha256(
            model.get_normalized_bases()
        ),
    }
    if mode == "random":
        protocol["seed"] = int(args.random_base_seed)
        protocol["aot_network"] = "trained_checkpoint_frozen"
    elif mode == "random_frozen":
        protocol["seed"] = int(model.config.base_seed)
        protocol["aot_network"] = (
            "jointly_trained_against_frozen_random_bases"
        )
    checkpoint = getattr(args, "evaluation_checkpoint", None)
    if checkpoint is not None:
        protocol["checkpoint"] = str(Path(checkpoint).resolve())
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SCGFM-ART cross-domain node classification on prepared "
            "node-centric PPR subgraphs."
        )
    )
    parser.add_argument(
        "--prepared-data-root",
        default=str(PROJECT_ROOT / "data" / "node_ppr_k400"),
    )
    parser.add_argument("--sources", default=",".join(SOURCE_NODE_DATASETS))
    parser.add_argument("--external-targets", default=",".join(EXTERNAL_NODE_TARGETS))
    parser.add_argument("--targets", help="Optional source-target subset.")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs" / "node")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-gin-layers", type=int, default=2)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--sinkhorn-temperature", type=float, default=0.1)
    parser.add_argument("--mean-temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--amp-dtype", choices=("bf16", "float16", "float32"), default="bf16"
    )
    parser.add_argument("--heatmap-interval", type=int, default=20)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--encoder-num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--random-projection-dim", type=int, default=256)
    parser.add_argument("--random-projection-seed", type=int, default=42)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--linear-epochs", type=int, default=1000)
    parser.add_argument("--linear-lr", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    parser.set_defaults(
        base_training="learned",
        base_seed=42,
        base_ema_decay=0.9,
        base_ema_eps=1e-6,
        base_ema_logit_eps=1e-4,
        base_ema_prior_weight=1.0,
        base_ema_assignment_temperature=0.05,
        base_ema_assignment_iterations=20,
        base_ema_uniform_floor=0.2,
        base_ema_topk=2,
        base_ema_capacity_leak=0.05,
        base_regularization="none",
        base_separation_weight=0.1,
        base_separation_margin=0.2,
        degree_normalization="max",
        coordinate_regularization="none",
        coord_variance_weight=1.0,
        coord_covariance_weight=1.0,
        coord_std_target=0.1,
        coord_eps=1e-4,
        coverage_volume_aux_weight=0.1,
        coordinate_volume_weight=0.01,
        coordinate_volume_eps=1e-4,
        coordinate_rank_weight=0.0,
        feature_readout="flatten",
        aot_weight_temperature=None,
        feature_transport="aot_mixture",
        evaluation_feature_mode="random_projection",
    )
    return parser.parse_args()

def _names(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        canonical_node_dataset_name(item)
        for item in value.split(",")
        if item.strip()
    ]


def lodo_training_sources(sources: list[str], target: str) -> list[str]:
    """Return source domains for one leakage-free LODO fold."""
    if target not in sources:
        raise ValueError(f"Unknown LODO target: {target}")
    return [name for name in sources if name != target]


def _model_config(args: argparse.Namespace) -> SCGFMARTConfig:
    return SCGFMARTConfig(
        K=args.K,
        M=args.M,
        base_parameterization="kernel",
        base_training=getattr(args, "base_training", "learned"),
        base_seed=getattr(args, "base_seed", 42),
        base_ema_decay=getattr(args, "base_ema_decay", 0.9),
        base_ema_eps=getattr(args, "base_ema_eps", 1e-6),
        base_ema_logit_eps=getattr(args, "base_ema_logit_eps", 1e-4),
        base_ema_prior_weight=getattr(args, "base_ema_prior_weight", 1.0),
        base_ema_assignment_temperature=getattr(
            args, "base_ema_assignment_temperature", 0.05
        ),
        base_ema_assignment_iterations=getattr(
            args, "base_ema_assignment_iterations", 20
        ),
        base_ema_uniform_floor=getattr(
            args, "base_ema_uniform_floor", 0.2
        ),
        base_ema_topk=getattr(args, "base_ema_topk", 2),
        base_ema_capacity_leak=getattr(
            args, "base_ema_capacity_leak", 0.05
        ),
        base_regularization=getattr(args, "base_regularization", "none"),
        base_separation_weight=getattr(
            args, "base_separation_weight", 0.1
        ),
        base_separation_margin=getattr(
            args, "base_separation_margin", 0.20
        ),
        hidden_dim=args.hidden_dim,
        num_gin_layers=args.num_gin_layers,
        degree_normalization=getattr(args, "degree_normalization", "max"),
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_temperature=args.sinkhorn_temperature,
        mean_temperature=args.mean_temperature,
        loss_variant="mean",
        coordinate_regularization=getattr(
            args, "coordinate_regularization", "none"
        ),
        coord_variance_weight=getattr(
            args, "coord_variance_weight", 1.0
        ),
        coord_covariance_weight=getattr(
            args, "coord_covariance_weight", 1.0
        ),
        coord_std_target=getattr(args, "coord_std_target", 0.1),
        coord_eps=getattr(args, "coord_eps", 1e-4),
        coverage_volume_aux_weight=getattr(
            args, "coverage_volume_aux_weight", 0.1
        ),
        coordinate_volume_weight=getattr(
            args, "coordinate_volume_weight", 0.01
        ),
        coordinate_volume_eps=getattr(
            args, "coordinate_volume_eps", 1e-4
        ),
        coordinate_rank_weight=getattr(
            args, "coordinate_rank_weight", 0.0
        ),
    )


def _effective_samples_per_class(args: argparse.Namespace) -> int:
    return 16 if args.smoke else int(args.samples_per_class)


def _open_dataset(
    args: argparse.Namespace,
    name: str,
    *,
    with_features: bool,
    apply_class_cap: bool = False,
):
    dataset = PreparedNodeSubgraphDataset(
        args.prepared_data_root,
        name,
        with_features=with_features,
    )
    if not apply_class_cap:
        return dataset
    samples_per_class = _effective_samples_per_class(args)
    if samples_per_class == 0:
        return dataset
    prepared_cap = int(dataset.manifest.get("samples_per_class", 0))
    if prepared_cap > 0 and samples_per_class >= prepared_cap:
        return dataset
    return Subset(
        dataset,
        class_cap_indices(
            dataset.labels,
            samples_per_class=samples_per_class,
            seed=args.seed,
        ),
    )


def _training_dataset(args: argparse.Namespace, names: list[str]):
    children = []
    for name in names:
        dataset = _open_dataset(
            args,
            name,
            with_features=False,
            apply_class_cap=False,
        )
        if (
            dataset.manifest.get("center_scope") != "all_nodes"
            or len(dataset) != int(dataset.manifest["num_nodes"])
        ):
            raise ValueError(
                f"Pretraining source {name} must use an all-node prepared "
                "cache; target-only per-class caches are not valid sources."
            )
        children.append(dataset)
    return ConcatDataset(children)


def _prepared_protocol(
    args: argparse.Namespace,
    names: list[str],
) -> dict[str, dict]:
    protocols = {}
    for name in names:
        dataset = PreparedNodeSubgraphDataset(
            args.prepared_data_root,
            name,
            with_features=False,
        )
        manifest = dataset.manifest
        protocols[name] = {
            key: manifest.get(key)
            for key in (
                "version",
                "num_nodes",
                "num_edges",
                "num_subgraphs",
                "max_subgraph_nodes",
                "ppr_method",
                "ppr_alpha",
                "ppr_eps",
                "max_pushes",
                "center_scope",
                "samples_per_class",
                "sampling_seed",
            )
            if manifest.get(key) is not None
        }
        protocols[name]["centers_sha256"] = manifest.get(
            "artifacts", {}
        ).get("centers", {}).get("sha256")
    return protocols


def _train_or_load(
    args: argparse.Namespace,
    source_names: list[str],
    directory: Path,
    device: torch.device,
):
    model_path = directory / "model.pt"
    if args.resume and model_path.exists():
        model, payload = load_checkpoint(model_path, device)
        if source_names:
            requested_protocol = _prepared_protocol(
                args, source_names
            )
            checkpoint_protocol = payload.get(
                "train_config", {}
            ).get("prepared_data_protocol")
            if checkpoint_protocol != requested_protocol:
                raise ValueError(
                    "Checkpoint PPR/data protocol mismatch at "
                    f"{model_path}. The baseline-compatible PPR cache "
                    "requires retraining in a new output directory."
                )
        requested_sampling = {"mode": "all_nodes"}
        checkpoint_sampling = payload.get("train_config", {}).get(
            "center_sampling"
        )
        if checkpoint_sampling != requested_sampling:
            raise ValueError(
                f"Checkpoint center-sampling mismatch at {model_path}: "
                f"checkpoint={checkpoint_sampling!r}, "
                f"requested={requested_sampling!r}. Use a new output "
                "directory and retrain."
            )
        checkpoint_config = model.config.to_dict()
        requested_config = _model_config(args).to_dict()
        differences = {
            key: {
                "checkpoint": checkpoint_config.get(key),
                "requested": requested_config.get(key),
            }
            for key in sorted(
                set(checkpoint_config) | set(requested_config)
            )
            if checkpoint_config.get(key) != requested_config.get(key)
        }
        protected_training_keys = {
            "base_training",
            "base_seed",
            "base_ema_decay",
            "base_ema_eps",
            "base_ema_logit_eps",
            "base_ema_prior_weight",
            "base_ema_assignment_temperature",
            "base_ema_assignment_iterations",
            "base_ema_uniform_floor",
            "base_ema_topk",
            "base_ema_capacity_leak",
            "base_ema_protocol",
            "base_regularization",
            "base_separation_weight",
            "base_separation_margin",
            "coordinate_regularization",
            "coord_variance_weight",
            "coord_covariance_weight",
            "coord_std_target",
            "coord_eps",
            "coverage_volume_aux_weight",
            "coordinate_volume_weight",
            "coordinate_volume_eps",
            "coordinate_rank_weight",
        }
        protected_differences = {
            key: value
            for key, value in differences.items()
            if key in protected_training_keys
        }
        if protected_differences:
            raise ValueError(
                f"Checkpoint training-objective mismatch at {model_path}: "
                f"{protected_differences}. Base-training and coordinate "
                "regularization variants must use different output "
                "directories."
            )
        print(
            f"Resume completed node pretraining: {model_path} "
            f"(epoch={payload.get('epoch', 0)})",
            flush=True,
        )
        if differences:
            print(
                "Using completed checkpoint model config; ignored "
                f"command-line model differences: {differences}",
                flush=True,
            )
        return model
    dataset = _training_dataset(args, source_names)
    sampler = ShardAwareSampler(dataset, seed=args.seed)
    train_config = {
        "epochs": 1 if args.smoke else args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "use_amp": not args.no_amp,
        "amp_dtype": args.amp_dtype,
        "resume": args.resume,
        "seed": args.seed,
        "datasets": source_names,
        "prepared_data_protocol": _prepared_protocol(
            args, source_names
        ),
        "center_sampling": {
            "mode": "all_nodes",
        },
        "domain_balanced": False,
        "heatmap_interval": (
            1 if args.smoke else args.heatmap_interval
        ),
        "_sampler": sampler,
    }
    model, _ = pretrain(
        dataset,
        _model_config(args),
        train_config,
        device,
        directory,
    )
    return model


def _base_dataset(dataset):
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def _active_prepared_indices(dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        return torch.as_tensor(dataset.indices).long()
    return torch.arange(len(dataset), dtype=torch.long)


def _representation_name(
    feature_readout: str,
    feature_transport: str = "aot_mixture",
    evaluation_feature_mode: str = "random_projection",
    random_projection_dim: int = 256,
    random_projection_seed: int = 42,
) -> str:
    prefix = (
        "aot_full_mixed_base_aot"
        if feature_transport == "mixed_base_aot"
        else "aot_full"
    )
    if evaluation_feature_mode == "random_projection":
        prefix = (
            f"{prefix}_rp{int(random_projection_dim)}"
            f"_seed{int(random_projection_seed)}"
        )
    if feature_readout == "flatten":
        return prefix
    if feature_readout == "pool":
        return f"{prefix}_pool"
    return f"{prefix}_target"


def _head_config(
    args: argparse.Namespace,
) -> dict[str, int | float | str]:
    return {
        "protocol": NODE_HEAD_PROTOCOL,
        "linear_max_iter": (
            min(5, args.linear_epochs)
            if args.smoke
            else args.linear_epochs
        ),
        "linear_lr": args.linear_lr,
    }


def _evaluate_target(
    args: argparse.Namespace,
    model,
    target: str,
    target_kind: str,
    pretraining_sources: list[str],
    directory: Path,
    device: torch.device,
) -> list[dict]:
    metrics_path = directory / "fewshot_metrics.json"
    head_config = _head_config(args)
    evaluation_feature_mode = getattr(
        args, "evaluation_feature_mode", "random_projection"
    )
    random_projection_dim = int(
        getattr(args, "random_projection_dim", 256)
    )
    random_projection_seed = int(
        getattr(args, "random_projection_seed", 42)
    )
    base_protocol = _evaluation_base_protocol(args, model)
    explicit_base_control = hasattr(args, "evaluation_base_mode")
    target_data_protocol = _prepared_protocol(args, [target])[target]
    target_sampling = {
        "mode": "per_class_cap",
        "samples_per_class": _effective_samples_per_class(args),
        "seed": int(args.seed),
    }
    if args.resume and metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            payload.get("evaluation_version") == NODE_EVALUATION_VERSION
            and payload.get("encoder_version")
            == SCGFMARTEncoder.version_for_configuration(
                "aot_full",
                args.feature_readout,
                args.feature_transport,
                evaluation_feature_mode,
                random_projection_dim,
                random_projection_seed,
            )
            and payload.get("head_config") == head_config
            and payload.get("feature_transport")
            == args.feature_transport
            and payload.get("prepared_data_protocol")
            == target_data_protocol
            and payload.get("center_sampling") == target_sampling
            and (
                payload.get("evaluation_bases") == base_protocol
                or (
                    not explicit_base_control
                    and payload.get("evaluation_bases") is None
                )
            )
        ):
            print(f"Resume completed node evaluation: {metrics_path}")
            return payload["rows"]

    directory.mkdir(parents=True, exist_ok=True)
    dataset = _open_dataset(
        args,
        target,
        with_features=True,
        apply_class_cap=True,
    )
    base = _base_dataset(dataset)
    active = _active_prepared_indices(dataset)
    active_labels = base.labels[active]
    splits_local = create_episode_splits(
        active_labels,
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_runs=2 if args.smoke else args.n_runs,
        seed=args.seed,
    )
    splits = []
    for split in splits_local:
        support = active[torch.as_tensor(split["support_indices"]).long()]
        query = active[torch.as_tensor(split["query_indices"]).long()]
        splits.append(
            {
                "run": int(split["run"]),
                "seed": int(split["seed"]),
                "support_indices": support,
                "query_indices": query,
                "support_node_ids": base.centers[support].clone(),
                "query_node_ids": base.centers[query].clone(),
            }
        )
    torch.save(splits, directory / "episode_splits.pt")
    union = episode_index_union(splits)
    union_dataset = Subset(base, union.tolist())
    encoder = SCGFMARTEncoder(
        model,
        device=device,
        max_dim=model.M,
        num_projections=1,
        top_k=args.top_k,
        coordinate_mode="aot_full",
        aot_weight_temperature=args.aot_weight_temperature,
        feature_readout=args.feature_readout,
        feature_transport=args.feature_transport,
        node_feature_mode=evaluation_feature_mode,
        random_projection_dim=random_projection_dim,
        random_projection_seed=random_projection_seed,
    )
    embeddings, _ = encoder.encode_dataset(
        union_dataset,
        batch_size=args.encoder_batch_size,
        num_workers=args.encoder_num_workers,
    )
    row_map = torch.full((len(base),), -1, dtype=torch.long)
    row_map[union] = torch.arange(union.numel())
    torch.save(
        {
            "embeddings": embeddings,
            "prepared_indices": union,
            "encoder_version": encoder.encoder_version,
            "feature_recoding": "N T^T X",
            "coordinate_mode": "aot_full",
            "feature_readout": args.feature_readout,
            "feature_transport": args.feature_transport,
            "evaluation_feature_mode": evaluation_feature_mode,
            "random_projection_dim": random_projection_dim,
            "random_projection_seed": random_projection_seed,
            "prepared_data_protocol": target_data_protocol,
            "encoder_config": encoder.output_config,
            "evaluation_bases": base_protocol,
            "pretraining_base_training": model.config.base_training,
            "pretraining_base_seed": model.config.base_seed,
            "pretraining_base_ema_decay": model.config.base_ema_decay,
            "base_regularization": model.config.base_regularization,
            "base_separation_weight": (
                model.config.base_separation_weight
            ),
            "base_separation_margin": (
                model.config.base_separation_margin
            ),
            "coordinate_regularization": (
                model.config.coordinate_regularization
            ),
        },
        directory / "embeddings_episode_union.pt",
    )
    torch.save(
        row_map, directory / "prepared_index_to_embedding_row.pt"
    )
    summary, episode_rows = evaluate_fewshot_splits(
        embeddings,
        base.labels,
        row_map,
        splits,
        device=device,
        linear_epochs=int(head_config["linear_max_iter"]),
        linear_lr=args.linear_lr,
        heads=("linear",),
    )
    enriched = [
        {
            "target": target,
            "target_kind": target_kind,
            **row,
        }
        for row in episode_rows
    ]
    write_csv(directory / "episode_results.csv", enriched)
    rows = []
    for head, values in summary["heads"].items():
        rows.append(
            {
                "target": target,
                "target_kind": target_kind,
                "pretraining_sources": "+".join(pretraining_sources),
                "head": head,
                "evaluation_base_mode": base_protocol["mode"],
                "evaluation_base_sha256": base_protocol[
                    "normalized_bases_sha256"
                ],
                "pretraining_base_training": model.config.base_training,
                "pretraining_base_seed": model.config.base_seed,
                "pretraining_base_ema_decay": (
                    model.config.base_ema_decay
                ),
                "coordinate_regularization": (
                    model.config.coordinate_regularization
                ),
                "evaluation_feature_mode": evaluation_feature_mode,
                "random_projection_dim": random_projection_dim,
                "random_projection_seed": random_projection_seed,
                **values,
            }
        )
    write_json(
        metrics_path,
        {
            "target": target,
            "target_kind": target_kind,
            "pretraining_sources": pretraining_sources,
            "full_node_pool_size": len(base),
            "active_pool_size": int(active.numel()),
            "encoded_episode_union_size": int(union.numel()),
            "embedding_dim": int(embeddings.shape[1]),
            "encoder_version": encoder.encoder_version,
            "coordinate_mode": "aot_full",
            "feature_readout": args.feature_readout,
            "feature_transport": args.feature_transport,
            "evaluation_feature_mode": evaluation_feature_mode,
            "random_projection_dim": random_projection_dim,
            "random_projection_seed": random_projection_seed,
            "prepared_data_protocol": target_data_protocol,
            "evaluation_version": NODE_EVALUATION_VERSION,
            "head_protocol": NODE_HEAD_PROTOCOL,
            "head_config": head_config,
            "center_sampling": target_sampling,
            "evaluation_bases": base_protocol,
            "pretraining_base_training": model.config.base_training,
            "pretraining_base_seed": model.config.base_seed,
            "pretraining_base_ema_decay": model.config.base_ema_decay,
            "coordinate_regularization": (
                model.config.coordinate_regularization
            ),
            "metrics": ["accuracy"],
            "summary": summary,
            "rows": rows,
        },
    )
    return rows


def _write_root_summary(
    output: Path,
    rows: list[dict],
    failures: list[dict],
    feature_readout: str,
    feature_transport: str,
    evaluation_feature_mode: str,
    random_projection_dim: int,
    random_projection_seed: int,
    base_training: str,
    base_seed: int,
    base_ema_decay: float,
    base_regularization: str,
    base_separation_weight: float,
    base_separation_margin: float,
    coordinate_regularization: str,
    coord_variance_weight: float,
    coord_covariance_weight: float,
    coord_std_target: float,
    coverage_volume_aux_weight: float,
    coordinate_volume_weight: float,
    coordinate_volume_eps: float,
    coordinate_rank_weight: float,
) -> None:
    suffix = _representation_name(
        feature_readout,
        feature_transport,
        evaluation_feature_mode,
        random_projection_dim,
        random_projection_seed,
    )
    write_csv(output / "all_results_long.csv", rows)
    write_csv(output / f"all_results_long_{suffix}.csv", rows)
    heads = ("linear",)
    targets = list(dict.fromkeys(row["target"] for row in rows))
    table = []
    for head in heads:
        row = {"head": head}
        for target in targets:
            match = next(
                (
                    item
                    for item in rows
                    if item["head"] == head and item["target"] == target
                ),
                None,
            )
            row[target] = (
                None if match is None else match["accuracy_mean"]
            )
        table.append(row)
    write_csv(output / "acc_table.csv", table)
    write_csv(output / f"acc_table_{suffix}.csv", table)
    write_csv(output / "failures.csv", failures)
    write_csv(output / f"failures_{suffix}.csv", failures)
    write_json(
        output / "experiment_summary.json",
        {
            "protocol": "ppr_all_source_nodes_target_class_cap_lodo_v3",
            "model": f"kernel_mean_{base_training}_{suffix}",
            "base_training": base_training,
            "base_seed": base_seed,
            "base_ema_decay": base_ema_decay,
            "base_regularization": base_regularization,
            "base_separation_weight": base_separation_weight,
            "base_separation_margin": base_separation_margin,
            "coordinate_regularization": coordinate_regularization,
            "coord_variance_weight": coord_variance_weight,
            "coord_covariance_weight": coord_covariance_weight,
            "coord_std_target": coord_std_target,
            "coverage_volume_aux_weight": coverage_volume_aux_weight,
            "coordinate_volume_weight": coordinate_volume_weight,
            "coordinate_volume_eps": coordinate_volume_eps,
            "coordinate_rank_weight": coordinate_rank_weight,
            "evaluation_version": NODE_EVALUATION_VERSION,
            "head_protocol": NODE_HEAD_PROTOCOL,
            "metrics": ["accuracy"],
            "feature_readout": feature_readout,
            "feature_transport": feature_transport,
            "evaluation_feature_mode": evaluation_feature_mode,
            "random_projection_dim": random_projection_dim,
            "random_projection_seed": random_projection_seed,
            "rows": rows,
            "failures": failures,
        },
    )
    write_json(
        output / f"experiment_summary_{suffix}.json",
        {
            "protocol": "ppr_all_source_nodes_target_class_cap_lodo_v3",
            "model": f"kernel_mean_{base_training}_{suffix}",
            "base_training": base_training,
            "base_seed": base_seed,
            "base_ema_decay": base_ema_decay,
            "base_regularization": base_regularization,
            "base_separation_weight": base_separation_weight,
            "base_separation_margin": base_separation_margin,
            "coordinate_regularization": coordinate_regularization,
            "coord_variance_weight": coord_variance_weight,
            "coord_covariance_weight": coord_covariance_weight,
            "coord_std_target": coord_std_target,
            "coverage_volume_aux_weight": coverage_volume_aux_weight,
            "coordinate_volume_weight": coordinate_volume_weight,
            "coordinate_volume_eps": coordinate_volume_eps,
            "coordinate_rank_weight": coordinate_rank_weight,
            "evaluation_version": NODE_EVALUATION_VERSION,
            "head_protocol": NODE_HEAD_PROTOCOL,
            "metrics": ["accuracy"],
            "feature_readout": feature_readout,
            "feature_transport": feature_transport,
            "evaluation_feature_mode": evaluation_feature_mode,
            "random_projection_dim": random_projection_dim,
            "random_projection_seed": random_projection_seed,
            "rows": rows,
            "failures": failures,
        },
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_workers = 0
        args.encoder_num_workers = 0
        args.batch_size = min(args.batch_size, 16)
        args.encoder_batch_size = min(args.encoder_batch_size, 16)
    set_seed(args.seed)
    device = resolve_device(args.device)
    sources = _names(args.sources)
    external = _names(args.external_targets)
    targets = _names(args.targets) or sources
    unknown = sorted(set(targets) - set(sources))
    if unknown:
        raise ValueError(f"LODO targets must be source domains: {unknown}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    representation = _representation_name(
        args.feature_readout,
        args.feature_transport,
        args.evaluation_feature_mode,
        args.random_projection_dim,
        args.random_projection_seed,
    )
    write_json(output / "run_config.json", vars(args))
    write_json(
        output
        / f"run_config_{representation}.json",
        vars(args),
    )
    print(
        "SCGFM-ART node LODO | "
        f"device={device} | sources={sources} | targets={targets} | "
        "bases=kernel objective=mean coordinates=aot_full "
        f"base_training={args.base_training} "
        f"base_regularization={args.base_regularization} "
        f"base_seed={args.base_seed} "
        f"base_ema_decay={args.base_ema_decay} "
        f"coordinate_regularization={args.coordinate_regularization} "
        f"feature_readout={args.feature_readout} "
        f"feature_transport={args.feature_transport} "
        f"evaluation_features={args.evaluation_feature_mode} "
        f"rp_dim={args.random_projection_dim} "
        f"rp_seed={args.random_projection_seed} "
        "pretrain_centers=all_nodes "
        "downstream_samples_per_class="
        f"{_effective_samples_per_class(args)}",
        flush=True,
    )

    rows: list[dict] = []
    failures: list[dict] = []
    for target in targets:
        fold = output / f"leaveout_{target}"
        training_sources = lodo_training_sources(sources, target)
        try:
            model = _train_or_load(
                args,
                training_sources,
                fold / "pretrain",
                device,
            )
            rows.extend(
                _evaluate_target(
                    args,
                    model,
                    target,
                    "lodo_source",
                    training_sources,
                    fold
                    / f"evaluation_{representation}",
                    device,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "target": target,
                    "stage": "lodo",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if args.fail_fast:
                raise

    if not args.skip_external and external:
        try:
            source_model = _train_or_load(
                args,
                sources,
                output / "source_pretrain",
                device,
            )
            for target in external:
                try:
                    rows.extend(
                        _evaluate_target(
                            args,
                            source_model,
                            target,
                            "external",
                            sources,
                            output
                            / f"target_{target}"
                            / f"evaluation_{representation}",
                            device,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "target": target,
                            "stage": "external_evaluation",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if args.fail_fast:
                        raise
        except Exception as exc:
            failures.append(
                {
                    "target": "+".join(external),
                    "stage": "source_pretrain",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if args.fail_fast:
                raise
    _write_root_summary(
        output,
        rows,
        failures,
        args.feature_readout,
        args.feature_transport,
        args.evaluation_feature_mode,
        args.random_projection_dim,
        args.random_projection_seed,
        args.base_training,
        args.base_seed,
        args.base_ema_decay,
        args.base_regularization,
        args.base_separation_weight,
        args.base_separation_margin,
        args.coordinate_regularization,
        args.coord_variance_weight,
        args.coord_covariance_weight,
        args.coord_std_target,
        args.coverage_volume_aux_weight,
        args.coordinate_volume_weight,
        args.coordinate_volume_eps,
        args.coordinate_rank_weight,
    )
    print(
        f"Completed node experiment: {output / 'experiment_summary.json'} "
        f"| rows={len(rows)} failures={len(failures)}",
        flush=True,
    )


if __name__ == "__main__":
    main()



