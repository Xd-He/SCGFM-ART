from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import torch
from torch_geometric.loader import DataLoader
from tqdm import trange

from .checkpoint import load_checkpoint, save_checkpoint
from .data import DomainBalancedSampler
from .model import SCGFMARTConfig, SCGFMARTModel
from .utils import write_csv, write_json
from .visualization import save_bases_heatmap


def pretrain(
    graphs,
    model_config: SCGFMARTConfig,
    train_config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
) -> tuple[SCGFMARTModel, list[dict[str, Any]]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    num_workers = int(train_config.get("num_workers", 0))
    sampler = train_config.get("_sampler")
    if sampler is None:
        sampler = (
            DomainBalancedSampler(
                graphs, seed=int(train_config.get("seed", 42))
            )
            if bool(train_config.get("domain_balanced", False))
            else None
        )
    loader_options = {
        "batch_size": int(train_config.get("batch_size", 64)),
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = int(
            train_config.get("prefetch_factor", 2)
        )
    loader = DataLoader(graphs, **loader_options)
    checkpoint_path = output / "checkpoint_last.pt"
    resume = bool(train_config.get("resume", False))
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resume_payload = None
    if resume and checkpoint_path.exists():
        model, resume_payload = load_checkpoint(checkpoint_path, device)
        if model.config.to_dict() != model_config.to_dict():
            saved = model.config.to_dict()
            requested = model_config.to_dict()
            differences = {
                key: {
                    "checkpoint": saved.get(key),
                    "requested": requested.get(key),
                }
                for key in sorted(set(saved) | set(requested))
                if saved.get(key) != requested.get(key)
            }
            raise ValueError(
                f"Checkpoint config mismatch at {checkpoint_path}: "
                f"{differences}. Resuming interrupted optimization requires "
                "the original model arguments."
            )
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        for saved_row in resume_payload.get("history", []):
            clean_row = {}
            for key, value in saved_row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    clean_row[key] = None
                else:
                    clean_row[key] = value
            history.append(clean_row)
        print(
            f"Resume interrupted pretraining: {checkpoint_path} "
            f"from epoch {start_epoch}",
            flush=True,
        )
    else:
        model = SCGFMARTModel(model_config, device=device).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("SCGFM-ART has no trainable network parameters.")
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(train_config.get("lr", 1e-3)),
        fused=device.type == "cuda",
    )
    if resume_payload and resume_payload.get("optimizer") is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    epochs = int(train_config.get("epochs", 60))
    use_amp = bool(train_config.get("use_amp", True)) and device.type == "cuda"
    requested_amp_dtype = str(
        train_config.get("amp_dtype", "bf16")
    ).lower()
    if not use_amp or requested_amp_dtype == "float32":
        amp_dtype = torch.float32
        autocast_enabled = False
    elif (
        requested_amp_dtype == "bf16"
        and torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.bfloat16
        autocast_enabled = True
    else:
        amp_dtype = torch.float16
        autocast_enabled = True
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and amp_dtype == torch.float16,
    )
    if (
        resume_payload
        and resume_payload.get("scaler")
        and scaler.is_enabled()
    ):
        scaler.load_state_dict(resume_payload["scaler"])
    grad_clip = float(train_config.get("grad_clip", 5.0))
    heatmap_interval = int(train_config.get("heatmap_interval", 20))
    epoch_callback = train_config.get("_epoch_callback")

    for epoch in trange(
        start_epoch,
        epochs + 1,
        desc="Pretrain SCGFM-ART",
        initial=max(0, start_epoch - 1),
        total=epochs,
    ):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        model.train()
        aggregate: dict[str, torch.Tensor] = {}
        batches = 0
        batch_graph_counts: list[int] = []
        nonfinite_grad_batches = 0
        optimizer_steps = 0
        for batch in loader:
            batch_graph_counts.append(int(batch.num_graphs))
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=autocast_enabled,
            ):
                loss, logs = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = [
                parameter.grad
                for parameter in trainable_parameters
                if parameter.grad is not None
            ]
            if gradients:
                grad_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                gradient.detach().float()
                            )
                            for gradient in gradients
                        ]
                    )
                )
            else:
                grad_norm = loss.detach().new_zeros(())
            gradients_are_finite = bool(
                torch.isfinite(loss.detach())
                & torch.isfinite(grad_norm)
            )
            if gradients_are_finite:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += 1
                logged_grad_norm = grad_norm.detach()
                if model.config.base_training == "ema_centroid":
                    ema_update_logs = (
                        model.apply_pending_ema_base_update()
                    )
                    logs.update(
                        {
                            key: value.detach()
                            for key, value in ema_update_logs.items()
                        }
                    )
            else:
                nonfinite_grad_batches += 1
                bad_names = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ]
                optimizer.zero_grad(set_to_none=True)
                model.clear_pending_ema_base_update()
                if scaler.is_enabled():
                    scaler.update(
                        new_scale=max(scaler.get_scale() / 2.0, 1.0)
                    )
                logged_grad_norm = loss.detach().new_zeros(())
                if nonfinite_grad_batches == 1:
                    print(
                        "WARNING: skipped a batch with non-finite "
                        f"loss/gradients; parameters={bad_names}",
                        flush=True,
                    )
            logs["grad_norm"] = logged_grad_norm
            for key, value in logs.items():
                detached = torch.as_tensor(value, device=device).detach()
                aggregate[key] = (
                    detached
                    if key not in aggregate
                    else aggregate[key] + detached
                )
            batches += 1
        row = {"epoch": float(epoch)}
        metric_names = list(aggregate)
        metric_values = (
            torch.stack(
                [aggregate[name] / max(1, batches) for name in metric_names]
            )
            .float()
            .cpu()
            .tolist()
        )
        row.update(dict(zip(metric_names, metric_values)))
        row["nonfinite_grad_batches"] = float(nonfinite_grad_batches)
        row["optimizer_steps"] = float(optimizer_steps)
        row["num_batches"] = float(batches)
        row["actual_batch_size_mean"] = float(
            sum(batch_graph_counts) / max(1, len(batch_graph_counts))
        )
        row["actual_batch_size_min"] = float(
            min(batch_graph_counts) if batch_graph_counts else 0
        )
        row["actual_batch_size_max"] = float(
            max(batch_graph_counts) if batch_graph_counts else 0
        )
        if "coordinate_rank_batch_valid" in aggregate:
            valid_batches = float(aggregate["coordinate_rank_batch_valid"].cpu())
            row["coordinate_rank_invalid_batches"] = float(
                max(0.0, batches - valid_batches)
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        row["epoch_seconds"] = epoch_seconds
        row["graphs_per_second"] = len(graphs) / max(
            epoch_seconds, 1e-8
        )
        row["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        if epoch_callback is not None:
            model.eval()
            callback_values = epoch_callback(model, epoch, dict(row))
            if callback_values:
                for key, value in callback_values.items():
                    if torch.is_tensor(value):
                        value = value.detach().float().cpu()
                        value = (
                            float(value)
                            if value.numel() == 1
                            else value.tolist()
                        )
                    row[f"diagnostic_{key}"] = value
        history.append(row)
        decomposed_text = ""
        if model.config.loss_variant == "decomposed":
            decomposed_text = (
                f" L_push={row['loss_pushforward']:.6f}"
                f" L_var={row['loss_variance']:.6f}"
                f" L_sel={row['loss_selection']:.6f}"
                f" W_push={row['weighted_pushforward']:.6f}"
                f" W_var={row['weighted_variance']:.6f}"
                f" W_sel={row['weighted_selection']:.6f} |"
            )
        elif model.config.loss_variant == "variance":
            decomposed_text = (
                f" L_var={row['loss_variance']:.6f}"
                f" V_mean={row['variance_mean']:.6f} |"
            )
        coordinate_text = ""
        if model.config.coordinate_regularization == "vicreg":
            coordinate_text = (
                f" L_cvar={row['loss_coord_variance']:.6f}"
                f" L_ccov={row['loss_coord_covariance']:.3e}"
                f" W_cvar={row['weighted_coord_variance']:.6f}"
                f" W_ccov={row['weighted_coord_covariance']:.3e}"
                f" Q_std={row['coord_std_mean']:.5f}"
                f" Q_cov={row['coord_cov_offdiag_rms']:.3e} |"
            )
        elif model.config.coordinate_regularization == "coverage_volume":
            coordinate_text = (
                f" L_cov={row['loss_coverage']:.6f}"
                f" V_logdet={row['coordinate_volume_logdet']:.5f}"
                f" L_vol={row['loss_coordinate_volume']:.5f}"
                f" L_cv={row['loss_coverage_volume_aux']:.6f}"
                f" W_cv={row['weighted_coverage_volume_aux']:.6f}"
                f" erank={row['coordinate_effective_rank']:.3f}"
                f" eig_min={row['coordinate_eigenvalue_min']:.3e}"
                f" eig_max={row['coordinate_eigenvalue_max']:.3e} |"
            )
        elif model.config.coordinate_regularization == "effective_rank":
            coordinate_text = (
                f" L_rank={row['loss_coordinate_effective_rank']:.6f}"
                f" W_rank={row['weighted_coordinate_effective_rank']:.6f}"
                f" erank={row['coordinate_effective_rank']:.3f}"
                f"/{row['coordinate_effective_rank_target']:.1f}"
                f" H_spec={row['coordinate_spectral_entropy']:.4f}"
                f" spec_mass={row['coordinate_spectral_mass']:.3e}"
                f" rank_cap={row['coordinate_rank_upper_bound']:.1f}"
                f" rank_valid={row['coordinate_rank_batch_valid']:.3f}"
                f" Qrel_std={row['coordinate_relative_std']:.3e} |"
            )
        ema_text = ""
        if model.config.base_training == "ema_centroid":
            ema_text = (
                f" EMA_dB={row['ema_base_change_rms']:.3e}"
                f" EMA_H={row['ema_assignment_entropy']:.4f}"
                f" EMA_UH={row['ema_usage_entropy']:.4f}"
                f" EMA_effK={row['ema_effective_bases']:.2f}"
                f" EMA_cap={row['ema_capacity_residual']:.2e}"
                f" EMA_pair=[{row['ema_base_pairwise_rms_min']:.3e},"
                f"{row['ema_base_pairwise_rms_mean']:.3e}]"
                f" EMA_batch_mass="
                f"[{row['ema_batch_mass_min']:.3f},"
                f"{row['ema_batch_mass_max']:.3f}]"
                f" EMA_mass="
                f"[{row['ema_running_mass_min']:.3f},"
                f"{row['ema_running_mass_max']:.3f}] |"
            )
        separation_text = ""
        if model.config.base_regularization == "separation":
            separation_text = (
                f" L_sep={row['loss_base_separation']:.6f}"
                f" W_sep={row['weighted_base_separation']:.6f}"
                f" B_pair=[{row['base_pairwise_distance_min']:.3e},"
                f"{row['base_pairwise_distance_mean']:.3e},"
                f"{row['base_pairwise_distance_median']:.3e}]"
                f" active={row['base_separation_active_pair_ratio']:.3f} |"
            )
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"L_mean={row['loss_mean']:.6f} | "
            f"L_total={row['total']:.6f} |"
            f"{decomposed_text} "
            f"{coordinate_text} "
            f"{ema_text} "
            f"{separation_text} "
            f"E_aot={row['aot_energy_mean']:.6f} "
            f"H_aot={row['aot_entropy']:.4f} "
            f"sinkhorn={row['sinkhorn_residual']:.2e} "
            f"grad={row['grad_norm']:.2e} | "
            f"steps={int(row['optimizer_steps'])}/{batches} "
            f"bad_grad={int(row['nonfinite_grad_batches'])} | "
            f"{row['graphs_per_second']:.1f} graph/s "
            f"GPU_peak={row['peak_gpu_memory_mb']:.0f} MiB"
        )
        if heatmap_interval > 0 and epoch % heatmap_interval == 0:
            heatmap_path = save_bases_heatmap(
                model.get_normalized_bases(),
                output / "base_heatmaps" / f"epoch_{epoch:04d}.png",
                epoch,
            )
            print(f"Saved geometric-base heatmap: {heatmap_path}", flush=True)
        serializable_train_config = {
            key: value
            for key, value in train_config.items()
            if not key.startswith("_")
        }
        save_checkpoint(
            model,
            output / "checkpoint_last.pt",
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            train_config=serializable_train_config,
            history=history,
        )
        write_json(output / "train_metrics.json", {"history": history})
        write_csv(output / "train_metrics.csv", history)

    serializable_train_config = {
        key: value
        for key, value in train_config.items()
        if not key.startswith("_")
    }
    save_checkpoint(
        model,
        output / "model.pt",
        optimizer=optimizer,
        scaler=scaler,
        epoch=epochs,
        train_config=serializable_train_config,
        history=history,
    )
    torch.save(
        model.get_normalized_bases().detach().cpu(),
        output / "learned_bases.pt",
    )
    if model.config.base_training == "random_frozen":
        torch.save(
            {
                "bases": model.get_normalized_bases().detach().cpu(),
                "base_parameterization": model.config.base_parameterization,
                "base_training": model.config.base_training,
                "base_seed": model.config.base_seed,
            },
            output / "random_frozen_bases.pt",
        )
    elif model.config.base_training == "ema_centroid":
        torch.save(
            {
                "bases": model.get_normalized_bases().detach().cpu(),
                "base_parameterization": model.config.base_parameterization,
                "base_training": model.config.base_training,
                "base_ema_decay": model.config.base_ema_decay,
                "base_ema_mass": model.ema_base_mass.detach().cpu(),
                "base_ema_sum": model.ema_base_sum.detach().cpu(),
                "base_ema_updates": int(model.ema_base_updates),
            },
            output / "ema_centroid_bases.pt",
        )
    return model, history


