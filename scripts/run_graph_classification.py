from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from scgfm_art.checkpoint import (
    load_checkpoint,
    model_config_from_checkpoint,
)
from scgfm_art.data import (
    EXTERNAL_TARGETS,
    SOURCE_DATASETS,
    load_graphs,
    set_domain_id,
    structure_only_copies,
)
from scgfm_art.encoder import SCGFMARTEncoder
from scgfm_art.fewshot import evaluate_fewshot
from scgfm_art.model import SCGFMARTConfig, SCGFMARTModel
from scgfm_art.training import pretrain
from scgfm_art.utils import (
    resolve_device,
    set_seed,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SCGFM-ART cross-domain graph classification: five source "
            "LODO folds and two unseen external targets."
        )
    )
    parser.add_argument("--sources", default=",".join(SOURCE_DATASETS))
    parser.add_argument("--external-targets", default=",".join(EXTERNAL_TARGETS))
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument(
        "--data-root", default=str(PROJECT_ROOT / "data" / "TUDataset")
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs" / "graph")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--max-per-class", type=int)
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
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.set_defaults(
        tau=0.1,
        base_parameterization="kernel",
        base_embedding_dim=8,
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
        loss_variant="mean",
        pushforward_weight=1.0,
        variance_weight=1.0,
        selection_weight=1.0,
        coordinate_regularization="none",
        coord_variance_weight=1.0,
        coord_covariance_weight=1.0,
        coord_std_target=0.1,
        coord_eps=1e-4,
        coverage_volume_aux_weight=0.1,
        coordinate_volume_weight=0.01,
        coordinate_volume_eps=1e-4,
        coordinate_rank_weight=0.0,
        no_domain_balance=False,
        encoder_max_dim=100,
        encoder_projections=200,
        allow_encoder_truncation=False,
        coordinate_mode="aot_full",
        aot_weight_temperature=None,
        feature_transport="aot_mixture",
        structure_only_downstream=False,
    )
    return parser.parse_args()

def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def lodo_training_sources(sources: list[str], target: str) -> list[str]:
    """Return source domains for one leakage-free LODO fold."""
    if target not in sources:
        raise ValueError(f"Unknown LODO target: {target}")
    return [name for name in sources if name != target]


def _evaluation_suffix(args: argparse.Namespace) -> str:
    suffix = args.coordinate_mode
    if args.feature_transport == "mixed_base_aot":
        suffix += "_mixed_base_aot"
    return suffix


def model_config(args: argparse.Namespace) -> SCGFMARTConfig:
    return SCGFMARTConfig(
        K=args.K,
        M=args.M,
        tau=args.tau,
        base_parameterization=args.base_parameterization,
        base_embedding_dim=args.base_embedding_dim,
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
        loss_variant=getattr(args, "loss_variant", "mean"),
        pushforward_weight=getattr(args, "pushforward_weight", 1.0),
        variance_weight=getattr(args, "variance_weight", 1.0),
        selection_weight=getattr(args, "selection_weight", 1.0),
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


def config_differences(
    checkpoint_config: dict,
    requested_config: dict,
) -> dict[str, dict]:
    keys = sorted(set(checkpoint_config) | set(requested_config))
    return {
        key: {
            "checkpoint": checkpoint_config.get(key),
            "requested": requested_config.get(key),
        }
        for key in keys
        if checkpoint_config.get(key) != requested_config.get(key)
    }


def adopt_resume_model_config(
    args: argparse.Namespace,
    output: Path,
) -> dict[str, dict]:
    model_paths = sorted(output.glob("*/pretrain/model.pt"))
    if not model_paths:
        return {}
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in model_paths
    ]
    reference = model_config_from_checkpoint(payloads[0]).to_dict()
    for path, payload in zip(model_paths[1:], payloads[1:]):
        candidate = model_config_from_checkpoint(payload).to_dict()
        if candidate != reference:
            raise ValueError(
                "Completed checkpoints in the same experiment directory "
                f"use inconsistent model configs: {model_paths[0]} versus "
                f"{path}. Use separate output directories."
            )
    requested = model_config(args).to_dict()
    differences = config_differences(reference, requested)
    protected_keys = (
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
    )
    protected = {
        key: differences[key] for key in protected_keys if key in differences
    }
    if protected:
        raise ValueError(
            "Completed checkpoints use a different base-training or "
            "coordinate-regularization protocol: "
            f"{json.dumps(protected, ensure_ascii=False)}. Use a separate "
            "output directory."
        )
    for key, value in reference.items():
        if hasattr(args, key):
            setattr(args, key, value)
    if differences:
        print(
            "Resume adopted model config from completed checkpoint "
            f"{model_paths[0]}: "
            f"{json.dumps(differences, ensure_ascii=False)}",
            flush=True,
        )
    return differences


def train_config(
    args: argparse.Namespace,
    source_names: list[str],
) -> dict:
    return {
        "epochs": args.epochs,
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
        "domain_balanced": not args.no_domain_balance,
        "heatmap_interval": args.heatmap_interval,
    }


def load_dataset(
    args: argparse.Namespace,
    name: str,
    domain_id: int,
):
    graphs = load_graphs(
        args.data_root,
        name,
        max_nodes=args.max_nodes,
        max_per_class=args.max_per_class,
        seed=args.seed,
        drop_node_features=False,
    )
    return set_domain_id(graphs, domain_id)


def train_or_load(
    args: argparse.Namespace,
    source_graphs,
    source_names: list[str],
    directory: Path,
    device: torch.device,
):
    model_path = directory / "model.pt"
    if args.resume and model_path.exists():
        model, payload = load_checkpoint(model_path, device)
        expected = model_config(args).to_dict()
        differences = config_differences(
            model.config.to_dict(), expected
        )
        if differences:
            print(
                f"Resume completed pretraining with checkpoint-owned "
                f"model config: {model_path}. Ignored requested model "
                f"differences: {json.dumps(differences, ensure_ascii=False)}",
                flush=True,
            )
        print(
            f"Resume completed pretraining: {model_path} "
            f"(epoch={payload['epoch']})",
            flush=True,
        )
        return model
    structural_graphs = structure_only_copies(source_graphs)
    model, _ = pretrain(
        structural_graphs,
        model_config(args),
        train_config(args, source_names),
        device,
        directory,
    )
    return model


def _read_completed_evaluation(
    path: Path,
    expected_encoder_version: str,
    expected_aot_weight_temperature: float,
    expected_base_training: str,
    expected_base_seed: int,
) -> list[dict] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved_version = payload.get("encoder_version")
    if saved_version != expected_encoder_version:
        print(
            "Recompute stale evaluation: "
            f"{path} (encoder={saved_version!r}, expected="
            f"{expected_encoder_version!r})",
            flush=True,
        )
        return None
    if (
        payload.get("coordinate_mode") == "aot_full"
        and payload.get("aot_weight_temperature")
        != expected_aot_weight_temperature
    ):
        print(
            "Recompute stale evaluation: "
            f"{path} (aot_weight_temperature="
            f"{payload.get('aot_weight_temperature')!r}, expected="
            f"{expected_aot_weight_temperature!r})",
            flush=True,
        )
        return None
    saved_base_training = payload.get("pretraining_base_training")
    saved_base_seed = payload.get("pretraining_base_seed")
    legacy_learned = (
        saved_base_training is None
        and expected_base_training == "learned"
    )
    if not legacy_learned and (
        saved_base_training != expected_base_training
        or saved_base_seed != expected_base_seed
    ):
        print(
            "Recompute stale evaluation: "
            f"{path} (base_training={saved_base_training!r}, "
            f"base_seed={saved_base_seed!r}; expected "
            f"{expected_base_training!r}/{expected_base_seed})",
            flush=True,
        )
        return None
    return payload.get("rows")


def evaluate_target(
    args: argparse.Namespace,
    model,
    graphs,
    target: str,
    target_kind: str,
    pretraining_sources: list[str],
    directory: Path,
    device: torch.device,
) -> list[dict]:
    metrics_path = directory / "fewshot_metrics.json"
    resolved_aot_weight_temperature = (
        model.config.mean_temperature
        if args.aot_weight_temperature is None
        else args.aot_weight_temperature
    )
    if args.resume:
        completed = _read_completed_evaluation(
            metrics_path,
            SCGFMARTEncoder.version_for_configuration(
                args.coordinate_mode,
                "flatten",
                args.feature_transport,
            ),
            resolved_aot_weight_temperature,
            model.config.base_training,
            model.config.base_seed,
        )
        if completed is not None:
            print(f"Resume completed evaluation: {metrics_path}", flush=True)
            return completed

    directory.mkdir(parents=True, exist_ok=True)
    evaluation_graphs = (
        structure_only_copies(graphs)
        if args.structure_only_downstream
        else graphs
    )
    maximum_nodes = max(int(graph.num_nodes) for graph in graphs)
    effective_encoder_dim = (
        args.encoder_max_dim
        if args.allow_encoder_truncation
        else max(args.encoder_max_dim, maximum_nodes)
    )
    encoder = SCGFMARTEncoder(
        model,
        tau=args.tau,
        device=device,
        max_dim=effective_encoder_dim,
        num_projections=args.encoder_projections,
        top_k=args.top_k,
        coordinate_mode=args.coordinate_mode,
        aot_weight_temperature=args.aot_weight_temperature,
        feature_transport=args.feature_transport,
    )
    embeddings, labels = encoder.encode_dataset(
        evaluation_graphs,
        batch_size=args.encoder_batch_size,
        num_workers=args.encoder_num_workers,
    )
    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "encoder_version": encoder.encoder_version,
            "feature_recoding": "N T^T X",
            "coordinate_mode": args.coordinate_mode,
            "aot_weight_temperature": encoder.aot_weight_temperature,
            "feature_transport": args.feature_transport,
            "encoder_config": encoder.output_config,
            "pretraining_objective": model.config.loss_variant,
            "pretraining_base_training": model.config.base_training,
            "pretraining_base_seed": model.config.base_seed,
            "coordinate_regularization": (
                model.config.coordinate_regularization
            ),
            "loss_weights": {
                "pushforward": model.config.pushforward_weight,
                "variance": model.config.variance_weight,
                "selection": model.config.selection_weight,
            },
        },
        directory / "embeddings.pt",
    )
    summary, episode_rows = evaluate_fewshot(
        embeddings,
        labels,
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_runs=args.n_runs,
        seed=args.seed,
        device=device,
    )
    enriched_episodes = [
        {
            "target": target,
            "target_kind": target_kind,
            "coordinate_mode": args.coordinate_mode,
            "aot_weight_temperature": encoder.aot_weight_temperature,
            "feature_transport": args.feature_transport,
            **row,
        }
        for row in episode_rows
    ]
    write_csv(directory / "episode_results.csv", enriched_episodes)
    rows = []
    for head, values in summary["heads"].items():
        rows.append(
            {
                "target": target,
                "target_kind": target_kind,
                "pretraining_sources": "+".join(pretraining_sources),
                "coordinate_mode": args.coordinate_mode,
                "aot_weight_temperature": encoder.aot_weight_temperature,
                "feature_transport": args.feature_transport,
                "pretraining_base_training": model.config.base_training,
                "pretraining_base_seed": model.config.base_seed,
                "coordinate_regularization": (
                    model.config.coordinate_regularization
                ),
                "head": head,
                **values,
            }
        )
    write_json(
        metrics_path,
        {
            "target": target,
            "target_kind": target_kind,
            "pretraining_sources": pretraining_sources,
            "num_graphs": len(graphs),
            "embedding_dim": int(embeddings.shape[1]),
            "encoder_version": encoder.encoder_version,
            "feature_recoding": "N T^T X",
            "coordinate_mode": args.coordinate_mode,
            "aot_weight_temperature": encoder.aot_weight_temperature,
            "feature_transport": args.feature_transport,
            "encoder_config": encoder.output_config,
            "pretraining_objective": model.config.loss_variant,
            "pretraining_base_training": model.config.base_training,
            "pretraining_base_seed": model.config.base_seed,
            "coordinate_regularization": (
                model.config.coordinate_regularization
            ),
            "loss_weights": {
                "pushforward": model.config.pushforward_weight,
                "variance": model.config.variance_weight,
                "selection": model.config.selection_weight,
            },
            "maximum_nodes": maximum_nodes,
            "effective_encoder_max_dim": effective_encoder_dim,
            "encoder_truncation": (
                args.allow_encoder_truncation
                and maximum_nodes > args.encoder_max_dim
            ),
            "protocol": {
                "k_shot": args.k_shot,
                "n_query": args.n_query,
                "n_runs": args.n_runs,
                "support_only_normalization": True,
            },
            "rows": rows,
        },
    )
    return rows


def write_summaries(
    output: Path,
    rows: list[dict],
    failures: list[dict],
    sources: list[str],
    externals: list[str],
    args: argparse.Namespace,
) -> None:
    target_order = sources + ([] if args.skip_external else externals)
    suffix = _evaluation_suffix(args)
    write_csv(output / "all_results_long.csv", rows)
    write_csv(output / f"all_results_long_{suffix}.csv", rows)
    heads = ("prototype",)
    acc_table = []
    for head in heads:
        selected = {
            row["target"]: row
            for row in rows
            if row["head"] == head
        }
        acc_table.append(
            {
                "head": head,
                **{
                    target: (
                        selected[target]["accuracy_mean"]
                        if target in selected
                        else None
                    )
                    for target in target_order
                },
            }
        )
    write_csv(output / "acc_table.csv", acc_table)
    write_csv(output / f"acc_table_{suffix}.csv", acc_table)
    write_csv(output / "failures.csv", failures)
    write_csv(output / f"failures_{suffix}.csv", failures)
    write_json(
        output / "experiment_summary.json",
        {
            "variant": SCGFMARTModel.implementation_version,
            "encoder_version": (
                SCGFMARTEncoder.version_for_configuration(
                    args.coordinate_mode,
                    "flatten",
                    args.feature_transport,
                )
            ),
            "feature_recoding": "N T^T X",
            "coordinate_mode": args.coordinate_mode,
            "feature_transport": args.feature_transport,
            "aot_weight_temperature": (
                args.mean_temperature
                if args.aot_weight_temperature is None
                else args.aot_weight_temperature
            ),
            "objective": args.loss_variant,
            "loss_weights": {
                "pushforward": args.pushforward_weight,
                "variance": args.variance_weight,
                "selection": args.selection_weight,
            },
            "base_parameterization": args.base_parameterization,
            "base_training": args.base_training,
            "base_seed": args.base_seed,
            "base_ema_decay": args.base_ema_decay,
            "base_regularization": args.base_regularization,
            "base_separation_weight": args.base_separation_weight,
            "base_separation_margin": args.base_separation_margin,
            "coordinate_regularization": args.coordinate_regularization,
            "coord_variance_weight": args.coord_variance_weight,
            "coord_covariance_weight": args.coord_covariance_weight,
            "coord_std_target": args.coord_std_target,
            "coverage_volume_aux_weight": (
                args.coverage_volume_aux_weight
            ),
            "coordinate_volume_weight": args.coordinate_volume_weight,
            "coordinate_volume_eps": args.coordinate_volume_eps,
            "coordinate_rank_weight": args.coordinate_rank_weight,
            "sources": sources,
            "external_targets": (
                [] if args.skip_external else externals
            ),
            "rows": rows,
            "failures": failures,
        },
    )
    write_json(
        output / f"experiment_summary_{suffix}.json",
        {
            "variant": SCGFMARTModel.implementation_version,
            "encoder_version": (
                SCGFMARTEncoder.version_for_configuration(
                    args.coordinate_mode,
                    "flatten",
                    args.feature_transport,
                )
            ),
            "feature_recoding": "N T^T X",
            "coordinate_mode": args.coordinate_mode,
            "feature_transport": args.feature_transport,
            "aot_weight_temperature": (
                args.mean_temperature
                if args.aot_weight_temperature is None
                else args.aot_weight_temperature
            ),
            "objective": args.loss_variant,
            "loss_weights": {
                "pushforward": args.pushforward_weight,
                "variance": args.variance_weight,
                "selection": args.selection_weight,
            },
            "base_parameterization": args.base_parameterization,
            "base_training": args.base_training,
            "base_seed": args.base_seed,
            "base_ema_decay": args.base_ema_decay,
            "base_regularization": args.base_regularization,
            "base_separation_weight": args.base_separation_weight,
            "base_separation_margin": args.base_separation_margin,
            "coordinate_regularization": args.coordinate_regularization,
            "coord_variance_weight": args.coord_variance_weight,
            "coord_covariance_weight": args.coord_covariance_weight,
            "coord_std_target": args.coord_std_target,
            "coverage_volume_aux_weight": (
                args.coverage_volume_aux_weight
            ),
            "coordinate_volume_weight": args.coordinate_volume_weight,
            "coordinate_volume_eps": args.coordinate_volume_eps,
            "coordinate_rank_weight": args.coordinate_rank_weight,
            "sources": sources,
            "external_targets": (
                [] if args.skip_external else externals
            ),
            "rows": rows,
            "failures": failures,
        },
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_per_class = 16
        args.epochs = 2
        args.K = 4
        args.M = 8
        args.hidden_dim = 16
        args.encoder_projections = 16
        args.sinkhorn_iterations = 10
        args.batch_size = 64
        args.encoder_batch_size = 64
        args.n_query = 5
        args.n_runs = 2
        args.heatmap_interval = 1

    sources = _names(args.sources)
    externals = _names(args.external_targets)
    if len(sources) < 2:
        raise ValueError("At least two source datasets are required for LODO.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        adopt_resume_model_config(args, output)
    write_json(output / "experiment_config.json", vars(args))

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    print(
        f"Seven-dataset SCGFM-ART | device={device} | "
        f"bases={args.base_parameterization} | "
        f"base_training={args.base_training} | "
        f"base_regularization={args.base_regularization} | "
        f"base_seed={args.base_seed} | "
        f"base_ema_decay={args.base_ema_decay} | "
        f"coordinate_regularization={args.coordinate_regularization} | "
        f"objective={args.loss_variant} | "
        f"coordinates={args.coordinate_mode} | "
        f"feature_transport={args.feature_transport} | "
        f"sources={sources} | externals={externals}",
        flush=True,
    )

    datasets = {
        name: load_dataset(args, name, domain_id)
        for domain_id, name in enumerate(sources)
    }
    rows: list[dict] = []
    failures: list[dict] = []

    def record_failure(stage: str, target: str, exc: Exception) -> None:
        failure = {
            "stage": stage,
            "target": target,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        failures.append(failure)
        write_summaries(
            output, rows, failures, sources, externals, args
        )
        if args.fail_fast:
            raise exc

    for target in sources:
        try:
            set_seed(args.seed)
            train_names = lodo_training_sources(sources, target)
            training_graphs = [
                graph for name in train_names for graph in datasets[name]
            ]
            fold = output / f"leaveout_{target.replace('-', '_')}"
            model = train_or_load(
                args,
                training_graphs,
                train_names,
                fold / "pretrain",
                device,
            )
            rows.extend(
                evaluate_target(
                    args,
                    model,
                    datasets[target],
                    target,
                    "lodo",
                    train_names,
                    fold / f"evaluation_{_evaluation_suffix(args)}",
                    device,
                )
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_summaries(
                output, rows, failures, sources, externals, args
            )
        except Exception as exc:
            record_failure("lodo", target, exc)

    if not args.skip_external:
        source_model = None
        try:
            set_seed(args.seed)
            all_source_graphs = [
                graph for name in sources for graph in datasets[name]
            ]
            source_model = train_or_load(
                args,
                all_source_graphs,
                sources,
                output / "source_pretrain",
                device,
            )
            for external_id, target in enumerate(externals, len(sources)):
                try:
                    target_graphs = load_dataset(
                        args, target, external_id
                    )
                    rows.extend(
                        evaluate_target(
                            args,
                            source_model,
                            target_graphs,
                            target,
                            "external",
                            sources,
                            output
                            / f"target_{target.replace('-', '_')}"
                            / f"evaluation_{_evaluation_suffix(args)}",
                            device,
                        )
                    )
                    del target_graphs
                    write_summaries(
                        output,
                        rows,
                        failures,
                        sources,
                        externals,
                        args,
                    )
                except Exception as exc:
                    record_failure("external", target, exc)
        except Exception as exc:
            record_failure("source_pretrain", "all_sources", exc)
        finally:
            if source_model is not None:
                del source_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_summaries(output, rows, failures, sources, externals, args)
    print(
        f"Completed: {output / 'experiment_summary.json'} | "
        f"rows={len(rows)} failures={len(failures)}",
        flush=True,
    )


if __name__ == "__main__":
    main()



