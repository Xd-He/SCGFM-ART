from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SCGFMARTConfig:
    K: int = 16
    M: int = 32
    tau: float = 0.1
    base_parameterization: str = "kernel"
    base_training: str = "learned"
    base_seed: int = 42
    base_ema_decay: float = 0.9
    base_ema_eps: float = 1e-6
    base_ema_logit_eps: float = 1e-4
    base_ema_prior_weight: float = 1.0
    base_ema_assignment_temperature: float = 0.05
    base_ema_assignment_iterations: int = 20
    base_ema_uniform_floor: float = 0.2
    base_ema_topk: int = 2
    base_ema_capacity_leak: float = 0.05
    base_ema_protocol: str = "capacity_sparse_v2"
    base_regularization: str = "none"
    base_separation_weight: float = 0.1
    base_separation_margin: float = 0.20
    base_embedding_dim: int = 8
    hidden_dim: int = 64
    num_gin_layers: int = 2
    degree_normalization: str = "max"
    sinkhorn_iterations: int = 20
    sinkhorn_temperature: float = 0.1
    sinkhorn_tolerance: float = 1e-4
    mean_temperature: float = 0.1
    mean_loss_weight: float = 1.0
    loss_variant: str = "mean"
    pushforward_weight: float = 1.0
    variance_weight: float = 1.0
    selection_weight: float = 1.0
    # Legacy coordinate regularizers remain loadable for checkpoint
    # compatibility. The paper/mainline model uses mean coverage only.
    coordinate_regularization: str = "none"
    coord_variance_weight: float = 1.0
    coord_covariance_weight: float = 1.0
    coord_std_target: float = 0.1
    coord_eps: float = 1e-4
    coverage_volume_aux_weight: float = 0.1
    coordinate_volume_weight: float = 0.01
    coordinate_volume_eps: float = 1e-4
    coordinate_rank_weight: float = 0.0
    coordinate_rank_target_fraction: float = 0.5
    base_sharing: str = "independent"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        release_modes = {
            "base_parameterization": (self.base_parameterization, "kernel"),
            "base_training": (self.base_training, "learned"),
            "base_regularization": (self.base_regularization, "none"),
            "base_sharing": (self.base_sharing, "independent"),
            "degree_normalization": (self.degree_normalization, "max"),
            "loss_variant": (self.loss_variant, "mean"),
            "coordinate_regularization": (
                self.coordinate_regularization,
                "none",
            ),
        }
        invalid = {
            name: value
            for name, (value, expected) in release_modes.items()
            if value != expected
        }
        if invalid:
            raise ValueError(
                "The public release supports only the paper mainline; "
                f"invalid options: {invalid}."
            )
        if self.K < 1 or self.M < 2:
            raise ValueError("K >= 1 and M >= 2 are required.")
        if self.hidden_dim < 1 or self.num_gin_layers < 1:
            raise ValueError("AOT encoder dimensions must be positive.")
        if self.degree_normalization not in {"max", "raw"}:
            raise ValueError(
                "degree_normalization must be 'max' or 'raw'."
            )
        if self.base_parameterization not in {"kernel", "metric"}:
            raise ValueError(
                "base_parameterization must be 'kernel' or 'metric'."
            )
        if self.base_sharing not in {"independent", "shared"}:
            raise ValueError(
                "base_sharing must be 'independent' or 'shared'."
            )
        if (
            self.base_sharing == "shared"
            and self.base_parameterization != "kernel"
        ):
            raise ValueError(
                "base_sharing='shared' is only defined for kernel bases."
            )
        if self.base_training not in {
            "learned",
            "random_frozen",
            "ema_centroid",
        }:
            raise ValueError(
                "base_training must be 'learned', 'random_frozen', or "
                "'ema_centroid'."
            )
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative.")
        if not 0 <= self.base_ema_decay < 1:
            raise ValueError("base_ema_decay must be in [0, 1).")
        if self.base_ema_eps <= 0:
            raise ValueError("base_ema_eps must be positive.")
        if not 0 < self.base_ema_logit_eps < 0.5:
            raise ValueError("base_ema_logit_eps must be in (0, 0.5).")
        if self.base_ema_prior_weight < 0:
            raise ValueError("base_ema_prior_weight must be non-negative.")
        if self.base_ema_assignment_temperature <= 0:
            raise ValueError(
                "base_ema_assignment_temperature must be positive."
            )
        if self.base_ema_assignment_iterations < 1:
            raise ValueError(
                "base_ema_assignment_iterations must be positive."
            )
        if not 0 <= self.base_ema_uniform_floor <= 1:
            raise ValueError("base_ema_uniform_floor must be in [0, 1].")
        if self.base_ema_topk < 0:
            raise ValueError("base_ema_topk must be non-negative.")
        if not 0 <= self.base_ema_capacity_leak <= 1:
            raise ValueError("base_ema_capacity_leak must be in [0, 1].")
        if self.base_ema_protocol not in {
            "softmax_v1",
            "capacity_sparse_v2",
        }:
            raise ValueError(
                "base_ema_protocol must be 'softmax_v1' or "
                "'capacity_sparse_v2'."
            )
        if self.base_regularization not in {"none", "separation"}:
            raise ValueError(
                "base_regularization must be 'none' or 'separation'."
            )
        if self.base_separation_weight < 0:
            raise ValueError(
                "base_separation_weight must be non-negative."
            )
        if self.base_separation_margin <= 0:
            raise ValueError(
                "base_separation_margin must be positive."
            )
        if (
            self.base_regularization == "separation"
            and self.base_training != "learned"
        ):
            raise ValueError(
                "base_regularization='separation' requires "
                "base_training='learned'."
            )
        if (
            self.base_training == "ema_centroid"
            and self.base_parameterization != "kernel"
        ):
            raise ValueError(
                "base_training='ema_centroid' currently requires the "
                "bounded symmetric kernel parameterization."
            )
        if self.base_embedding_dim < 1:
            raise ValueError("base_embedding_dim must be positive.")
        if self.sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive.")
        if self.tau <= 0 or self.sinkhorn_temperature <= 0 or self.mean_temperature <= 0:
            raise ValueError("All temperatures must be positive.")
        if self.loss_variant not in {"mean", "decomposed", "variance"}:
            raise ValueError(
                "loss_variant must be 'mean', 'decomposed', or 'variance'."
            )
        if self.coordinate_regularization not in {
            "none",
            "vicreg",
            "coverage_volume",
            "effective_rank",
        }:
            raise ValueError(
                "coordinate_regularization must be 'none', 'vicreg', "
                "'coverage_volume', or 'effective_rank'."
            )
        if (
            self.coordinate_regularization in {"vicreg", "coverage_volume"}
            and self.base_training in {"random_frozen", "ema_centroid"}
        ):
            raise ValueError(
                "Response-space regularization uses stop-gradient transports "
                "to shape geometric bases, so it cannot be combined with "
                "gradient-trained bases. It cannot be combined with "
                "base_training='random_frozen' or 'ema_centroid'."
            )
        if (
            self.coordinate_regularization in {
                "coverage_volume",
                "effective_rank",
            }
            and self.loss_variant == "variance"
        ):
            raise ValueError(
                "The selected coordinate regularization requires "
                "differentiable full GW energies and cannot be combined "
                "with loss_variant='variance'."
            )
        if self.coord_variance_weight < 0 or self.coord_covariance_weight < 0:
            raise ValueError("Coordinate VICReg weights must be non-negative.")
        if self.coord_std_target <= 0 or self.coord_eps <= 0:
            raise ValueError(
                "coord_std_target and coord_eps must be positive."
            )
        if self.coverage_volume_aux_weight < 0:
            raise ValueError(
                "coverage_volume_aux_weight must be non-negative."
            )
        if self.coordinate_volume_weight < 0:
            raise ValueError(
                "coordinate_volume_weight must be non-negative."
            )
        if self.coordinate_volume_eps <= 0:
            raise ValueError("coordinate_volume_eps must be positive.")
        if self.coordinate_rank_weight < 0:
            raise ValueError("coordinate_rank_weight must be non-negative.")
        if self.mean_loss_weight < 0:
            raise ValueError("mean_loss_weight must be non-negative.")
        if not 0 < self.coordinate_rank_target_fraction <= 1:
            raise ValueError(
                "coordinate_rank_target_fraction must be in (0, 1]."
            )
        if self.base_sharing == "shared" and self.base_training != "learned":
            raise ValueError(
                "base_sharing='shared' requires base_training='learned'."
            )
        loss_weights = (
            self.pushforward_weight,
            self.variance_weight,
            self.selection_weight,
        )
        if any(weight < 0 for weight in loss_weights):
            raise ValueError("Decomposed loss weights must be non-negative.")
        if self.loss_variant == "decomposed" and not any(
            weight > 0 for weight in loss_weights
        ):
            raise ValueError(
                "At least one decomposed loss weight must be positive."
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SparseGraphBatch:
    """Ragged binary graph batch without an ``N x N`` allocation."""

    edge_index: torch.Tensor
    adjacency: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    counts: torch.Tensor
    local_index: torch.Tensor
    node_mask: torch.Tensor
    degree: torch.Tensor
    measure: torch.Tensor
    num_graphs: int
    max_num_nodes: int


class GINLayer(nn.Module):
    """GIN layer with sparse production and dense-oracle aggregation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        updated = self.mlp((1.0 + self.eps) * hidden + adjacency @ hidden)
        output = self.norm(hidden + updated)
        if node_mask is not None:
            output = output * node_mask[..., None]
        return output

    def forward_sparse(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Exact binary-adjacency GIN aggregation over directed edges."""
        aggregated = torch.zeros_like(hidden)
        if edge_index.numel() > 0:
            source, target = edge_index
            aggregated.index_add_(
                0,
                source,
                hidden.index_select(0, target),
            )
        updated = self.mlp((1.0 + self.eps) * hidden + aggregated)
        return self.norm(hidden + updated)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class SCGFMARTModel(nn.Module):
    """SCGFM-ART with mean or explicitly decomposed coverage."""

    implementation_version = "scgfm_art_release_v1"
    compatible_checkpoint_versions = frozenset({implementation_version})

    def __init__(
        self,
        config: SCGFMARTConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or SCGFMARTConfig()
        cfg = self.config
        base_generator = None
        if cfg.base_training == "random_frozen":
            base_generator = torch.Generator(device=torch.device(device))
            base_generator.manual_seed(int(cfg.base_seed))
        train_bases = cfg.base_training == "learned"

        def initialize_base(shape: tuple[int, ...]) -> torch.Tensor:
            if base_generator is None:
                return torch.randn(*shape, device=device)
            value = torch.randn(
                *shape, device=device, generator=base_generator
            )
            # Match the global RNG consumption of the learned-base model so
            # the AOT network has the same initialization in a controlled
            # learned-versus-frozen comparison.
            torch.randn(*shape, device=device)
            return value

        parameter_k = 1 if cfg.base_sharing == "shared" else cfg.K
        if cfg.base_parameterization == "kernel":
            self.bases_param = nn.Parameter(
                initialize_base((parameter_k, cfg.M, cfg.M)),
                requires_grad=train_bases,
            )
            self.register_parameter("base_coordinates", None)
        else:
            self.register_parameter("bases_param", None)
            self.base_coordinates = nn.Parameter(
                initialize_base(
                    (cfg.K, cfg.M, cfg.base_embedding_dim)
                ),
                requires_grad=train_bases,
            )

        # Current AOT architecture: normalized-degree-only graph GIN, shared
        # base row encoder, cross-attention logits, and marginal Sinkhorn.
        self.graph_input = nn.Linear(1, cfg.hidden_dim)
        self.graph_layers = nn.ModuleList(
            [GINLayer(cfg.hidden_dim) for _ in range(cfg.num_gin_layers)]
        )
        self.base_encoder = _mlp(cfg.M, cfg.hidden_dim, cfg.hidden_dim)
        self.query_projection = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.key_projection = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        if cfg.base_training == "ema_centroid":
            self.register_buffer(
                "ema_base_mass",
                torch.zeros(cfg.K, device=device, dtype=torch.float32),
            )
            self.register_buffer(
                "ema_base_sum",
                torch.zeros(
                    cfg.K,
                    cfg.M,
                    cfg.M,
                    device=device,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "ema_base_updates",
                torch.zeros((), device=device, dtype=torch.long),
            )
        else:
            self.register_buffer("ema_base_mass", None)
            self.register_buffer("ema_base_sum", None)
            self.register_buffer("ema_base_updates", None)
        self._pending_ema_base_mass: torch.Tensor | None = None
        self._pending_ema_base_sum: torch.Tensor | None = None
        self.to(device)

    @property
    def device(self) -> torch.device:
        parameter = (
            self.bases_param
            if self.bases_param is not None
            else self.base_coordinates
        )
        return parameter.device

    @property
    def K(self) -> int:
        return self.config.K

    @property
    def M(self) -> int:
        return self.config.M

    @property
    def tau(self) -> float:
        return self.config.tau

    @property
    def bases_trainable(self) -> bool:
        return self.config.base_training == "learned"

    @torch.no_grad()
    def ema_capacity_assignment(
        self, energies: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sparse capacity-aware responsibilities for the EMA E-step.

        Rows are graph responsibilities and sum to one. A dense entropic
        transport projection first matches a smoothed historical usage prior;
        optional row-wise top-k sparsification then makes the relational
        centroid update genuinely base-specific instead of a common mean.
        """
        if energies.ndim != 2 or energies.shape[1] != self.K:
            raise ValueError("energies must have shape [batch_size, K].")
        batch_size = energies.shape[0]
        if batch_size < 1:
            raise ValueError("EMA assignment requires a non-empty batch.")

        if self.config.base_ema_protocol == "softmax_v1":
            assignment = F.softmax(
                -energies.detach().float()
                / self.config.mean_temperature,
                dim=-1,
            )
            prior = assignment.mean(dim=0)
            return assignment, prior, assignment.new_zeros(())

        dtype = torch.float32
        device = energies.device
        uniform = torch.full(
            (self.K,), 1.0 / self.K, device=device, dtype=dtype
        )
        if self.ema_base_mass is None:
            historical_usage = uniform
        else:
            running_mass = self.ema_base_mass.detach().to(
                device=device, dtype=dtype
            )
            mass_total = running_mass.sum()
            historical_usage = torch.where(
                mass_total > self.config.base_ema_eps,
                running_mass / mass_total.clamp_min(
                    self.config.base_ema_eps
                ),
                uniform,
            )
        floor = self.config.base_ema_uniform_floor
        target_prior = (1.0 - floor) * historical_usage + floor * uniform
        target_prior = target_prior / target_prior.sum()

        log_assignment = (
            -energies.detach().to(dtype)
            / self.config.base_ema_assignment_temperature
        )
        log_column_mass = (
            target_prior.clamp_min(self.config.base_ema_eps).log()
            + math.log(float(batch_size))
        )
        for _ in range(self.config.base_ema_assignment_iterations):
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=1, keepdim=True
            )
            log_assignment = log_assignment + (
                log_column_mass
                - torch.logsumexp(log_assignment, dim=0)
            )[None, :]
        log_assignment = log_assignment - torch.logsumexp(
            log_assignment, dim=1, keepdim=True
        )
        dense_assignment = log_assignment.exp()

        topk = min(self.config.base_ema_topk, self.K)
        if 0 < topk < self.K:
            values, indices = torch.topk(
                dense_assignment, k=topk, dim=1, sorted=False
            )
            assignment = torch.zeros_like(dense_assignment)
            assignment.scatter_(1, indices, values)
            assignment = assignment / assignment.sum(
                dim=1, keepdim=True
            ).clamp_min(self.config.base_ema_eps)
            leak = self.config.base_ema_capacity_leak
            if leak > 0:
                # A small full-support component makes the sparse proposal
                # feasible for every target capacity. Reprojection restores
                # column usage after top-k without returning to the diffuse
                # common-mean responsibilities used by the old variant.
                assignment = (
                    (1.0 - leak) * assignment
                    + leak * target_prior[None, :]
                )
                log_assignment = assignment.clamp_min(
                    self.config.base_ema_eps
                ).log()
                for _ in range(
                    self.config.base_ema_assignment_iterations
                ):
                    log_assignment = log_assignment - torch.logsumexp(
                        log_assignment, dim=1, keepdim=True
                    )
                    log_assignment = log_assignment + (
                        log_column_mass
                        - torch.logsumexp(log_assignment, dim=0)
                    )[None, :]
                log_assignment = log_assignment - torch.logsumexp(
                    log_assignment, dim=1, keepdim=True
                )
                assignment = log_assignment.exp()
        else:
            assignment = dense_assignment

        usage = assignment.mean(dim=0)
        capacity_residual = (usage - target_prior).abs().max()
        return assignment, target_prior, capacity_residual

    @torch.no_grad()
    def stage_ema_base_update(
        self,
        adjacency: torch.Tensor,
        transport: torch.Tensor,
        energies: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build detached sufficient statistics for the relational M-step."""
        if self.config.base_training != "ema_centroid":
            return {}
        assignment, target_prior, capacity_residual = (
            self.ema_capacity_assignment(energies)
        )
        transport = transport.detach().float()
        adjacency = adjacency.detach().float()
        adjacency_transport = torch.einsum(
            "bij,bkjm->bkim", adjacency, transport
        )
        transported_relation = torch.einsum(
            "bkna,bknc->bkac", transport, adjacency_transport
        )
        pushforward = (self.M**2) * transported_relation
        batch_mass = assignment.sum(dim=0)
        batch_sum = torch.einsum(
            "bk,bkmn->kmn", assignment, pushforward
        )
        self._pending_ema_base_mass = batch_mass
        self._pending_ema_base_sum = batch_sum
        entropy = -torch.sum(
            assignment
            * torch.log(assignment.clamp_min(self.config.eps)),
            dim=-1,
        ).mean()
        usage = batch_mass / batch_mass.sum().clamp_min(
            self.config.base_ema_eps
        )
        usage_entropy = -torch.sum(
            usage * torch.log(usage.clamp_min(self.config.eps))
        )
        return {
            "ema_assignment_entropy": entropy,
            "ema_usage_entropy": usage_entropy,
            "ema_effective_bases": usage_entropy.exp(),
            "ema_capacity_residual": capacity_residual,
            "ema_target_prior_min": target_prior.min(),
            "ema_target_prior_max": target_prior.max(),
            "ema_batch_mass_min": batch_mass.min(),
            "ema_batch_mass_max": batch_mass.max(),
        }

    @torch.no_grad()
    def stage_ema_base_update_sparse(
        self,
        graph: SparseGraphBatch,
        transport: torch.Tensor,
        energies: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Sparse equivalent of the relational EMA sufficient statistics."""
        if self.config.base_training != "ema_centroid":
            return {}
        assignment, target_prior, capacity_residual = (
            self.ema_capacity_assignment(energies)
        )
        transported_relation, _ = self.sparse_transported_relation(
            graph,
            transport.detach().float(),
        )
        pushforward = (self.M**2) * transported_relation
        batch_mass = assignment.sum(dim=0)
        batch_sum = torch.einsum(
            "bk,bkmn->kmn",
            assignment,
            pushforward,
        )
        self._pending_ema_base_mass = batch_mass
        self._pending_ema_base_sum = batch_sum
        entropy = -torch.sum(
            assignment
            * torch.log(assignment.clamp_min(self.config.eps)),
            dim=-1,
        ).mean()
        usage = batch_mass / batch_mass.sum().clamp_min(
            self.config.base_ema_eps
        )
        usage_entropy = -torch.sum(
            usage * torch.log(usage.clamp_min(self.config.eps))
        )
        return {
            "ema_assignment_entropy": entropy,
            "ema_usage_entropy": usage_entropy,
            "ema_effective_bases": usage_entropy.exp(),
            "ema_capacity_residual": capacity_residual,
            "ema_target_prior_min": target_prior.min(),
            "ema_target_prior_max": target_prior.max(),
            "ema_batch_mass_min": batch_mass.min(),
            "ema_batch_mass_max": batch_mass.max(),
        }

    @torch.no_grad()
    def apply_pending_ema_base_update(self) -> dict[str, torch.Tensor]:
        """Apply one projected EMA closed-form centroid update."""
        if self.config.base_training != "ema_centroid":
            return {}
        if (
            self._pending_ema_base_mass is None
            or self._pending_ema_base_sum is None
        ):
            raise RuntimeError("No pending EMA base statistics to apply.")
        if self.ema_base_mass is None or self.ema_base_sum is None:
            raise RuntimeError("EMA base buffers are missing.")
        decay = self.config.base_ema_decay
        previous = self.get_normalized_bases().detach()
        prior_initialized = previous.new_zeros(())
        if (
            int(self.ema_base_updates.item()) == 0
            and float(self.ema_base_mass.sum().item())
            <= self.config.base_ema_eps
            and self.config.base_ema_prior_weight > 0
        ):
            # A one-batch-equivalent symmetric pseudo-count retains the
            # randomly initialized atlas. Consequently the first centroid
            # update is an actual EMA interpolation instead of cancelling
            # (1 - decay) in numerator and denominator and overwriting B_0.
            prior_mass = (
                self.config.base_ema_prior_weight
                * self._pending_ema_base_mass.sum()
                / float(self.K)
            )
            self.ema_base_mass.fill_(prior_mass)
            self.ema_base_sum.copy_(prior_mass * previous)
            prior_initialized = previous.new_ones(())
        self.ema_base_mass.mul_(decay).add_(
            self._pending_ema_base_mass, alpha=1.0 - decay
        )
        self.ema_base_sum.mul_(decay).add_(
            self._pending_ema_base_sum, alpha=1.0 - decay
        )
        centroid = self.ema_base_sum / (
            self.ema_base_mass + self.config.base_ema_eps
        )[:, None, None]
        centroid = 0.5 * (
            centroid + centroid.transpose(-1, -2)
        )
        centroid = centroid.clamp(0.0, 1.0)
        eye = torch.eye(
            self.M, device=self.device, dtype=centroid.dtype
        )
        centroid = centroid * (1.0 - eye)
        valid = self.ema_base_mass > self.config.base_ema_eps
        centroid = torch.where(
            valid[:, None, None], centroid, previous
        )

        bounded = centroid.clamp(
            self.config.base_ema_logit_eps,
            1.0 - self.config.base_ema_logit_eps,
        )
        logits = torch.logit(bounded)
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        self.bases_param.copy_(logits)
        updated = self.get_normalized_bases().detach()
        change = torch.sqrt((updated - previous).square().mean())
        if self.K > 1:
            pairwise_rms = torch.pdist(
                updated.reshape(self.K, -1), p=2
            ) / math.sqrt(float(self.M * self.M))
            pairwise_mean = pairwise_rms.mean()
            pairwise_min = pairwise_rms.min()
        else:
            pairwise_mean = updated.new_zeros(())
            pairwise_min = updated.new_zeros(())
        self.ema_base_updates.add_(1)
        self.clear_pending_ema_base_update()
        return {
            "ema_base_change_rms": change,
            "ema_running_mass_min": self.ema_base_mass.min(),
            "ema_running_mass_max": self.ema_base_mass.max(),
            "ema_update_count": self.ema_base_updates.float(),
            "ema_prior_initialized": prior_initialized,
            "ema_base_pairwise_rms_mean": pairwise_mean,
            "ema_base_pairwise_rms_min": pairwise_min,
        }

    def clear_pending_ema_base_update(self) -> None:
        self._pending_ema_base_mass = None
        self._pending_ema_base_sum = None

    def get_normalized_bases(self) -> torch.Tensor:
        if self.config.base_parameterization == "metric":
            distances = torch.cdist(
                self.base_coordinates, self.base_coordinates, p=2
            )
            scale = distances.amax(
                dim=(-1, -2), keepdim=True
            ).clamp_min(self.config.eps)
            bases = distances / scale
            eye = torch.eye(
                self.M, device=self.device, dtype=bases.dtype
            )
            return bases * (1.0 - eye)

        symmetric = 0.5 * (
            self.bases_param + self.bases_param.transpose(-1, -2)
        )
        bases = torch.sigmoid(symmetric)
        eye = torch.eye(self.M, device=self.device, dtype=bases.dtype)
        bases = bases * (1.0 - eye)
        if self.config.base_sharing == "shared":
            bases = bases.expand(self.K, -1, -1)
        return bases

    @torch.no_grad()
    def reinitialize_bases(self, seed: int) -> torch.Tensor:
        """Reset only the geometric bases and preserve the trained AOT net.

        The distribution is exactly the one used by ``__init__`` for the
        selected base parameterization.  A device-local generator keeps the
        operation deterministic without changing the global RNG state used
        later by the few-shot classifiers.
        """
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        parameter = (
            self.bases_param
            if self.bases_param is not None
            else self.base_coordinates
        )
        parameter.normal_(mean=0.0, std=1.0, generator=generator)
        return self.get_normalized_bases().detach()

    @staticmethod
    def clean_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = (adjacency > 0).float()
        if adjacency.numel() == 0:
            return adjacency
        eye = torch.eye(
            adjacency.shape[-1],
            device=adjacency.device,
            dtype=adjacency.dtype,
        )
        return adjacency * (1.0 - eye)

    @staticmethod
    def clean_edge_index(
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Remove self-loops and duplicate directed edges on the device.

        This is exactly equivalent to ``clean_adjacency(to_dense_adj(...))``
        for the unweighted edge lists consumed by the model.
        """
        edge_index = edge_index.long()
        if edge_index.numel() == 0 or num_nodes == 0:
            return edge_index.new_empty((2, 0))
        source, target = edge_index
        valid = (
            (source >= 0)
            & (target >= 0)
            & (source < num_nodes)
            & (target < num_nodes)
            & (source != target)
        )
        source = source[valid]
        target = target[valid]
        if source.numel() == 0:
            return edge_index.new_empty((2, 0))
        linear = torch.unique(
            source * int(num_nodes) + target,
            sorted=True,
        )
        source = torch.div(
            linear,
            int(num_nodes),
            rounding_mode="floor",
        )
        target = torch.remainder(linear, int(num_nodes))
        return torch.stack((source, target), dim=0)

    def prepare_sparse_batch(self, batch_data) -> SparseGraphBatch:
        """Build degree, measure, and ragged indices without dense adjacency."""
        node_batch = batch_data.batch.long()
        num_nodes = int(node_batch.numel())
        if hasattr(batch_data, "ptr"):
            ptr = batch_data.ptr.long()
            counts = ptr[1:] - ptr[:-1]
        else:
            num_graphs = (
                int(node_batch.max().item()) + 1
                if num_nodes > 0
                else 0
            )
            counts = torch.bincount(
                node_batch,
                minlength=num_graphs,
            )
            ptr = torch.cat(
                [counts.new_zeros(1), counts.cumsum(dim=0)]
            )
        num_graphs = int(counts.numel())
        max_num_nodes = (
            int(counts.max().item()) if num_graphs > 0 else 0
        )
        edge_index = self.clean_edge_index(
            batch_data.edge_index,
            num_nodes,
        )
        adjacency = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(
                edge_index.shape[1],
                device=node_batch.device,
                dtype=torch.float32,
            ),
            (num_nodes, num_nodes),
            device=node_batch.device,
            dtype=torch.float32,
        ).coalesce()
        degree = torch.zeros(
            num_nodes,
            device=node_batch.device,
            dtype=torch.float32,
        )
        if edge_index.numel() > 0:
            degree.index_add_(
                0,
                edge_index[0],
                torch.ones(
                    edge_index.shape[1],
                    device=node_batch.device,
                    dtype=torch.float32,
                ),
            )

        graph_degree = torch.zeros(
            num_graphs,
            device=node_batch.device,
            dtype=torch.float32,
        )
        graph_mass = torch.zeros_like(graph_degree)
        mass = degree + self.config.eps
        if num_nodes > 0:
            graph_degree.index_add_(0, node_batch, degree)
            graph_mass.index_add_(0, node_batch, mass)
        uniform = 1.0 / counts.clamp_min(1).float()
        normalized = mass / graph_mass.index_select(
            0, node_batch
        ).clamp_min(self.config.eps)
        measure = torch.where(
            graph_degree.index_select(0, node_batch) > self.config.eps,
            normalized,
            uniform.index_select(0, node_batch),
        )

        global_index = torch.arange(
            num_nodes,
            device=node_batch.device,
        )
        local_index = global_index - ptr.index_select(0, node_batch)
        node_mask = (
            torch.arange(
                max_num_nodes,
                device=node_batch.device,
            )[None]
            < counts[:, None]
        )
        return SparseGraphBatch(
            edge_index=edge_index,
            adjacency=adjacency,
            batch=node_batch,
            ptr=ptr,
            counts=counts,
            local_index=local_index,
            node_mask=node_mask,
            degree=degree,
            measure=measure,
            num_graphs=num_graphs,
            max_num_nodes=max_num_nodes,
        )

    @staticmethod
    def pad_sparse_nodes(
        values: torch.Tensor,
        graph: SparseGraphBatch,
    ) -> torch.Tensor:
        """Pad node-linear values; never allocates a pairwise node tensor."""
        output = values.new_zeros(
            (
                graph.num_graphs,
                graph.max_num_nodes,
                *values.shape[1:],
            )
        )
        if values.shape[0] > 0:
            output[graph.batch, graph.local_index] = values
        return output

    def encode_graph_nodes_sparse(
        self,
        graph: SparseGraphBatch,
    ) -> torch.Tensor:
        """Encode all ragged nodes using exact sparse GIN aggregation."""
        degree = graph.degree[:, None]
        if self.config.degree_normalization == "max":
            maximum = torch.zeros(
                graph.num_graphs,
                device=degree.device,
                dtype=degree.dtype,
            )
            if degree.numel() > 0:
                maximum.scatter_reduce_(
                    0,
                    graph.batch,
                    graph.degree,
                    reduce="amax",
                    include_self=True,
                )
            degree = degree / maximum.index_select(
                0, graph.batch
            )[:, None].clamp_min(1.0)
        hidden = F.relu(self.graph_input(degree))
        for layer in self.graph_layers:
            hidden = layer.forward_sparse(hidden, graph.edge_index)
        return hidden

    def graph_measure(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze = adjacency.dim() == 2
        if squeeze:
            adjacency = adjacency[None]
        if node_mask is None:
            node_mask = torch.ones(
                adjacency.shape[:2],
                device=adjacency.device,
                dtype=torch.bool,
            )
        elif node_mask.dim() == 1:
            node_mask = node_mask[None]
        degree = adjacency.sum(dim=-1)
        mass = (degree.float() + self.config.eps) * node_mask
        degree_sum = (degree * node_mask).sum(dim=-1, keepdim=True)
        uniform = node_mask.float() / node_mask.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        normalized = mass / mass.sum(dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )
        result = torch.where(
            degree_sum > self.config.eps, normalized, uniform
        )
        return result[0] if squeeze else result

    def encode_graph_nodes(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The default node input is graph-wise max-normalized undirected degree.
        degree = adjacency.sum(dim=-1, keepdim=True)
        if node_mask is not None:
            degree = degree * node_mask[..., None]
        if self.config.degree_normalization == "max":
            maximum = degree.amax(dim=-2, keepdim=True).clamp_min(1.0)
            degree = degree / maximum
        hidden = F.relu(self.graph_input(degree))
        if node_mask is not None:
            hidden = hidden * node_mask[..., None]
        for layer in self.graph_layers:
            hidden = layer(hidden, adjacency, node_mask)
        return hidden

    def encode_base_nodes(self, bases: torch.Tensor) -> torch.Tensor:
        return self.base_encoder(bases)

    def sinkhorn(
        self,
        logits: torch.Tensor,
        measure: torch.Tensor,
        base_measure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        squeeze = logits.dim() == 3
        if squeeze:
            logits = logits[None]
            measure = measure[None]
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_plan = torch.nan_to_num(
                logits.float() / cfg.sinkhorn_temperature,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ).clamp(-20.0, 20.0)
            log_plan = log_plan - log_plan.amax(dim=(-1, -2), keepdim=True)
            log_mu = torch.log(
                measure.float().clamp_min(cfg.eps)
            )[:, None, :, None]
            log_nu = torch.log(
                base_measure.float().clamp_min(cfg.eps)
            )[None, None, None, :]
            for _ in range(cfg.sinkhorn_iterations):
                log_plan = (
                    log_plan
                    + log_mu
                    - torch.logsumexp(log_plan, dim=-1, keepdim=True)
                )
                log_plan = (
                    log_plan
                    + log_nu
                    - torch.logsumexp(log_plan, dim=-2, keepdim=True)
                )
            transport = torch.exp(log_plan)

            # One differentiable transport-polytope rounding step removes the
            # finite-iteration marginal error.
            row_scale = torch.minimum(
                measure.float()[:, None]
                / transport.sum(dim=-1).clamp_min(cfg.eps),
                torch.ones_like(transport.sum(dim=-1)),
            )
            transport = transport * row_scale[..., None]
            col_scale = torch.minimum(
                base_measure.float()[None, None]
                / transport.sum(dim=-2).clamp_min(cfg.eps),
                torch.ones_like(transport.sum(dim=-2)),
            )
            transport = transport * col_scale[:, :, None, :]
            row_deficit = (
                measure.float()[:, None] - transport.sum(dim=-1)
            ).clamp_min(0.0)
            col_deficit = (
                base_measure.float()[None, None] - transport.sum(dim=-2)
            ).clamp_min(0.0)
            missing = row_deficit.sum(dim=-1).clamp_min(cfg.eps)
            transport = transport + (
                row_deficit[..., None]
                * col_deficit[..., None, :]
                / missing[..., None, None]
            )
            row_error = torch.sum(
                torch.abs(
                    transport.sum(dim=-1) - measure.float()[:, None]
                ),
                dim=-1,
            )
            col_error = torch.sum(
                torch.abs(
                    transport.sum(dim=-2)
                    - base_measure.float()[None, None]
                ),
                dim=-1,
            )
            residual = torch.maximum(row_error, col_error)
        if squeeze:
            return transport[0], residual[0]
        return transport, residual

    def predict_transport(
        self,
        adjacency: torch.Tensor,
        bases: torch.Tensor | None = None,
        base_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transport, measure, residual = self.predict_transport_batch(
            adjacency[None],
            torch.ones(
                (1, adjacency.shape[0]),
                device=adjacency.device,
                dtype=torch.bool,
            ),
            bases=bases,
            base_context=base_context,
        )
        return transport[0], measure[0], residual[0]

    def predict_transport_batch(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        bases: torch.Tensor | None = None,
        base_context: torch.Tensor | None = None,
        graph_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adjacency = self.clean_adjacency(adjacency)
        adjacency = adjacency * node_mask[:, :, None] * node_mask[:, None, :]
        bases = self.get_normalized_bases() if bases is None else bases
        if graph_context is None:
            graph_context = self.encode_graph_nodes(adjacency, node_mask)
        if base_context is None:
            base_context = self.encode_base_nodes(bases)
        query = self.query_projection(graph_context)
        key = self.key_projection(base_context)
        logits = torch.einsum("bnh,kmh->bknm", query, key) / math.sqrt(
            self.config.hidden_dim
        )
        measure = self.graph_measure(adjacency, node_mask)
        base_measure = torch.full(
            (self.M,),
            1.0 / self.M,
            device=adjacency.device,
            dtype=torch.float32,
        )
        transport, residual = self.sinkhorn(logits, measure, base_measure)
        return transport, measure, residual

    def predict_transport_to_graph_bases_batch(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        bases: torch.Tensor,
        graph_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict one AOT coupling to one graph-specific basis per graph.

        ``bases`` has shape ``[batch, M, M]``. This differs from
        :meth:`predict_transport_batch`, whose ``[K, M, M]`` bases are
        shared by every graph in the batch.
        """
        if bases.dim() != 3 or bases.shape[0] != adjacency.shape[0]:
            raise ValueError(
                "Expected graph-specific bases with shape [batch, M, M]."
            )
        if bases.shape[1:] != (self.M, self.M):
            raise ValueError(
                f"Expected bases with trailing shape ({self.M}, {self.M})."
            )
        adjacency = self.clean_adjacency(adjacency)
        adjacency = adjacency * node_mask[:, :, None] * node_mask[:, None, :]
        if graph_context is None:
            graph_context = self.encode_graph_nodes(adjacency, node_mask)
        base_context = self.encode_base_nodes(bases)
        query = self.query_projection(graph_context)
        key = self.key_projection(base_context)
        logits = torch.einsum(
            "bnh,bmh->bnm", query, key
        ) / math.sqrt(self.config.hidden_dim)
        measure = self.graph_measure(adjacency, node_mask)
        base_measure = torch.full(
            (self.M,),
            1.0 / self.M,
            device=adjacency.device,
            dtype=torch.float32,
        )
        transport, residual = self.sinkhorn(
            logits[:, None], measure, base_measure
        )
        return transport[:, 0], measure, residual[:, 0]

    def predict_transport_sparse_batch(
        self,
        graph: SparseGraphBatch,
        bases: torch.Tensor | None = None,
        base_context: torch.Tensor | None = None,
        graph_context: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Predict shared-base couplings from a ragged sparse graph batch."""
        bases = self.get_normalized_bases() if bases is None else bases
        if graph_context is None:
            graph_context = self.pad_sparse_nodes(
                self.encode_graph_nodes_sparse(graph),
                graph,
            )
        if base_context is None:
            base_context = self.encode_base_nodes(bases)
        query = self.query_projection(graph_context)
        key = self.key_projection(base_context)
        logits = torch.einsum(
            "bnh,kmh->bknm", query, key
        ) / math.sqrt(self.config.hidden_dim)
        measure = self.pad_sparse_nodes(graph.measure, graph)
        base_measure = torch.full(
            (self.M,),
            1.0 / self.M,
            device=logits.device,
            dtype=torch.float32,
        )
        transport, residual = self.sinkhorn(
            logits,
            measure,
            base_measure,
        )
        return transport, measure, residual, graph_context

    def predict_transport_to_graph_bases_sparse_batch(
        self,
        graph: SparseGraphBatch,
        bases: torch.Tensor,
        graph_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict one sparse-graph coupling to one base per graph."""
        if bases.dim() != 3 or bases.shape[0] != graph.num_graphs:
            raise ValueError(
                "Expected graph-specific bases with shape [batch, M, M]."
            )
        if bases.shape[1:] != (self.M, self.M):
            raise ValueError(
                f"Expected bases with trailing shape ({self.M}, {self.M})."
            )
        if graph_context is None:
            graph_context = self.pad_sparse_nodes(
                self.encode_graph_nodes_sparse(graph),
                graph,
            )
        base_context = self.encode_base_nodes(bases)
        query = self.query_projection(graph_context)
        key = self.key_projection(base_context)
        logits = torch.einsum(
            "bnh,bmh->bnm", query, key
        ) / math.sqrt(self.config.hidden_dim)
        measure = self.pad_sparse_nodes(graph.measure, graph)
        base_measure = torch.full(
            (self.M,),
            1.0 / self.M,
            device=logits.device,
            dtype=torch.float32,
        )
        transport, residual = self.sinkhorn(
            logits[:, None],
            measure,
            base_measure,
        )
        return transport[:, 0], measure, residual[:, 0]

    def sparse_adjacency_transport(
        self,
        graph: SparseGraphBatch,
        transport: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``A T`` exactly from edges for padded couplings."""
        flat_transport = transport[
            graph.batch,
            :,
            graph.local_index,
            :,
        ]
        if (
            graph.edge_index.numel() > 0
            and flat_transport.shape[0] >= 2048
        ):
            flat_result = torch.sparse.mm(
                graph.adjacency,
                flat_transport.reshape(flat_transport.shape[0], -1),
            ).reshape_as(flat_transport)
        elif graph.edge_index.numel() > 0:
            source, target = graph.edge_index
            flat_result = torch.zeros_like(flat_transport)
            flat_result.index_add_(
                0,
                source,
                flat_transport.index_select(0, target),
            )
        else:
            flat_result = torch.zeros_like(flat_transport)
        result = torch.zeros_like(transport)
        if flat_result.shape[0] > 0:
            result[
                graph.batch,
                :,
                graph.local_index,
                :,
            ] = flat_result
        return result

    def sparse_graph_constant(
        self,
        graph: SparseGraphBatch,
        measure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``sum_ij A_ij^2 mu_i mu_j`` over binary edges."""
        measure = graph.measure if measure is None else measure
        output = measure.new_zeros(graph.num_graphs)
        if graph.edge_index.numel() > 0:
            source, target = graph.edge_index
            contribution = measure.index_select(
                0, source
            ) * measure.index_select(0, target)
            output.index_add_(
                0,
                graph.batch.index_select(0, source),
                contribution,
            )
        return output

    def aot_gw_energy_sparse(
        self,
        graph: SparseGraphBatch,
        bases: torch.Tensor,
        transport: torch.Tensor,
        measure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact GW energy for a binary sparse relation; no ``N x N`` tensor."""
        with torch.autocast(
            device_type=transport.device.type,
            enabled=False,
        ):
            bases = bases.float()
            transport = transport.float()
            flat_measure = (
                graph.measure.float()
                if measure is None
                else measure.float()
            )
            nu = torch.full(
                (self.M,),
                1.0 / self.M,
                device=transport.device,
                dtype=torch.float32,
            )
            graph_constant = self.sparse_graph_constant(
                graph,
                flat_measure,
            )
            base_constant = torch.sum(
                bases.square()
                * nu[None, :, None]
                * nu[None, None, :],
                dim=(-1, -2),
            )
            adjacency_transport = self.sparse_adjacency_transport(
                graph,
                transport,
            )
            transport_base = torch.einsum(
                "bknm,kma->bkna",
                transport,
                bases,
            )
            cross = torch.sum(
                adjacency_transport * transport_base,
                dim=(-1, -2),
            )
            return (
                graph_constant[:, None]
                + base_constant[None]
                - 2.0 * cross
            ).clamp_min(0.0)

    def sparse_transported_relation(
        self,
        graph: SparseGraphBatch,
        transport: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``T^T A T`` and ``A T`` using the sparse binary relation."""
        adjacency_transport = self.sparse_adjacency_transport(
            graph,
            transport,
        )
        transported_relation = torch.einsum(
            "bkna,bknc->bkac",
            transport,
            adjacency_transport,
        )
        return transported_relation, adjacency_transport

    def aot_gw_energy_decomposition_sparse(
        self,
        graph: SparseGraphBatch,
        bases: torch.Tensor,
        transport: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact pushforward decomposition for the sparse binary relation."""
        with torch.autocast(
            device_type=transport.device.type,
            enabled=False,
        ):
            bases = bases.float()
            transport = transport.float()
            transported_relation, _ = self.sparse_transported_relation(
                graph,
                transport,
            )
            pushforward_mean = (self.M**2) * transported_relation
            # The cleaned input relation is binary, hence A^2 = A.
            pushforward_second = pushforward_mean
            pair_weight = 1.0 / float(self.M**2)
            mismatch = torch.sum(
                (bases[None] - pushforward_mean).square(),
                dim=(-1, -2),
            ) * pair_weight
            conditional_variance = torch.sum(
                (
                    pushforward_second
                    - pushforward_mean.square()
                ).clamp_min(0.0),
                dim=(-1, -2),
            ) * pair_weight
            return mismatch, conditional_variance

    def compact_conditional_variance_sparse(
        self,
        graph: SparseGraphBatch,
        bases: torch.Tensor,
        transport: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse equivalent of :meth:`compact_conditional_variance`."""
        with torch.autocast(
            device_type=transport.device.type,
            enabled=False,
        ):
            bases = bases.float()
            transport = transport.float()
            graph_constant = self.sparse_graph_constant(graph)
            transported_relation, adjacency_transport = (
                self.sparse_transported_relation(graph, transport)
            )
            captured_relation = (self.M**2) * torch.sum(
                transported_relation.square(),
                dim=(-1, -2),
            )
            conditional_variance = (
                graph_constant.detach()[:, None] - captured_relation
            )

            with torch.no_grad():
                nu = torch.full(
                    (self.M,),
                    1.0 / self.M,
                    device=transport.device,
                    dtype=torch.float32,
                )
                base_constant = torch.sum(
                    bases.square()
                    * nu[None, :, None]
                    * nu[None, None, :],
                    dim=(-1, -2),
                )
                transport_base = torch.einsum(
                    "bknm,kma->bkna",
                    transport.detach(),
                    bases.detach(),
                )
                cross = torch.sum(
                    adjacency_transport.detach() * transport_base,
                    dim=(-1, -2),
                )
                routing_energy = (
                    graph_constant[:, None]
                    + base_constant[None]
                    - 2.0 * cross
                ).clamp_min(0.0)
            return conditional_variance, routing_energy

    def aot_gw_energy(
        self,
        adjacency: torch.Tensor,
        bases: torch.Tensor,
        transport: torch.Tensor,
        measure: torch.Tensor,
    ) -> torch.Tensor:
        squeeze = adjacency.dim() == 2
        if squeeze:
            adjacency = adjacency[None]
            transport = transport[None]
            measure = measure[None]
        with torch.autocast(device_type=adjacency.device.type, enabled=False):
            adjacency = adjacency.float()
            bases = bases.float()
            transport = transport.float()
            measure = measure.float()
            nu = torch.full(
                (self.M,),
                1.0 / self.M,
                device=adjacency.device,
                dtype=torch.float32,
            )
            graph_constant = torch.sum(
                adjacency.square()
                * measure[:, :, None]
                * measure[:, None, :],
                dim=(-1, -2),
            )
            base_constant = torch.sum(
                bases.square() * nu[None, :, None] * nu[None, None, :],
                dim=(-1, -2),
            )
            adjacency_transport = torch.einsum(
                "bij,bkjm->bkim", adjacency, transport
            )
            transport_base = torch.einsum(
                "bknm,kma->bkna", transport, bases
            )
            cross = torch.sum(
                adjacency_transport * transport_base, dim=(-1, -2)
            )
            energy = (
                graph_constant[:, None] + base_constant[None] - 2.0 * cross
            ).clamp_min(0.0)
            return energy[0] if squeeze else energy

    def aot_gw_energy_decomposition(
        self,
        adjacency: torch.Tensor,
        bases: torch.Tensor,
        transport: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pushforward mismatch and conditional compression variance.

        Their sum is exactly the squared GW energy for the supplied
        coupling, up to floating-point roundoff.
        """
        squeeze = adjacency.dim() == 2
        if squeeze:
            adjacency = adjacency[None]
            transport = transport[None]
        with torch.autocast(device_type=adjacency.device.type, enabled=False):
            adjacency = adjacency.float()
            bases = bases.float()
            transport = transport.float()
            nu = torch.full(
                (self.M,),
                1.0 / self.M,
                device=adjacency.device,
                dtype=torch.float32,
            )
            conditional = (
                transport.transpose(-1, -2)
                / nu[None, None, :, None]
            )
            pushforward_mean = torch.einsum(
                "bkai,bij,bkcj->bkac",
                conditional,
                adjacency,
                conditional,
            )
            pushforward_second = torch.einsum(
                "bkai,bij,bkcj->bkac",
                conditional,
                adjacency.square(),
                conditional,
            )
            pair_measure = nu[:, None] * nu[None, :]
            mismatch = torch.sum(
                (bases[None] - pushforward_mean).square()
                * pair_measure[None, None],
                dim=(-1, -2),
            )
            conditional_variance = torch.sum(
                (
                    pushforward_second
                    - pushforward_mean.square()
                ).clamp_min(0.0)
                * pair_measure[None, None],
                dim=(-1, -2),
            )
        if squeeze:
            return mismatch[0], conditional_variance[0]
        return mismatch, conditional_variance

    def compact_conditional_variance(
        self,
        adjacency: torch.Tensor,
        bases: torch.Tensor,
        transport: torch.Tensor,
        measure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compact conditional variance and detached routing energy.

        For the uniform base measure, the conditional compression variance is

        ``c(A) - M^2 ||T^T A T||_F^2``.

        The full GW energy used for graph-to-base routing is computed from the
        same ``A T`` product and detached. Therefore only the compact variance
        contributes gradients, while routing still uses the complete energy.
        """
        squeeze = adjacency.dim() == 2
        if squeeze:
            adjacency = adjacency[None]
            transport = transport[None]
            measure = measure[None]
        with torch.autocast(device_type=adjacency.device.type, enabled=False):
            adjacency = adjacency.float()
            bases = bases.float()
            transport = transport.float()
            measure = measure.float()
            nu = torch.full(
                (self.M,),
                1.0 / self.M,
                device=adjacency.device,
                dtype=torch.float32,
            )
            graph_constant = torch.sum(
                adjacency.square()
                * measure[:, :, None]
                * measure[:, None, :],
                dim=(-1, -2),
            )
            adjacency_transport = torch.einsum(
                "bij,bkjm->bkim", adjacency, transport
            )
            transported_relation = torch.einsum(
                "bkna,bknc->bkac",
                transport,
                adjacency_transport,
            )
            captured_relation = (self.M**2) * torch.sum(
                transported_relation.square(), dim=(-1, -2)
            )
            conditional_variance = (
                graph_constant.detach()[:, None] - captured_relation
            )

            # Routing must use full GW energy, but it must not introduce
            # gradients from either R_k or the responsibility itself.
            with torch.no_grad():
                base_constant = torch.sum(
                    bases.square()
                    * nu[None, :, None]
                    * nu[None, None, :],
                    dim=(-1, -2),
                )
                transport_base = torch.einsum(
                    "bknm,kma->bkna",
                    transport.detach(),
                    bases.detach(),
                )
                cross = torch.sum(
                    adjacency_transport.detach() * transport_base,
                    dim=(-1, -2),
                )
                routing_energy = (
                    graph_constant[:, None]
                    + base_constant[None]
                    - 2.0 * cross
                ).clamp_min(0.0)
        if squeeze:
            return conditional_variance[0], routing_energy[0]
        return conditional_variance, routing_energy

    def mean_coverage(self, energies: torch.Tensor) -> torch.Tensor:
        temperature = self.config.mean_temperature
        # log-mean-exp is the normalized soft-min. It has the same model
        # gradients as log-sum-exp but remains on the non-negative GW scale.
        return -temperature * (
            torch.logsumexp(-energies / temperature, dim=-1)
            - math.log(self.K)
        )

    def base_separation_regularization(
        self, bases: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Margin repulsion on unique off-diagonal base-kernel entries."""
        if self.K < 2:
            zero = bases.float().sum() * 0.0
            return {
                "loss": zero,
                "distance_min": zero.detach(),
                "distance_mean": zero.detach(),
                "distance_median": zero.detach(),
                "active_pair_ratio": zero.detach(),
            }
        upper = torch.triu_indices(
            self.M,
            self.M,
            offset=1,
            device=bases.device,
        )
        vectors = bases.float()[:, upper[0], upper[1]]
        distances = torch.pdist(vectors, p=2) / math.sqrt(
            float(vectors.shape[1])
        )
        margin_violation = F.relu(
            self.config.base_separation_margin - distances
        )
        loss = margin_violation.square().mean()
        return {
            "loss": loss,
            "distance_min": distances.min().detach(),
            "distance_mean": distances.mean().detach(),
            "distance_median": distances.median().detach(),
            "active_pair_ratio": (
                distances < self.config.base_separation_margin
            ).float().mean().detach(),
        }

    def mean_coverage_decomposition(
        self,
        energies: torch.Tensor,
        pushforward_mismatch: torch.Tensor,
        compression_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Variational decomposition of normalized soft-min coverage.

        Returns responsibility-weighted mismatch, responsibility-weighted
        variance, and tau * KL(responsibility || uniform).
        """
        temperature = self.config.mean_temperature
        responsibility = F.softmax(-energies / temperature, dim=-1)
        mismatch_term = torch.sum(
            responsibility * pushforward_mismatch, dim=-1
        )
        variance_term = torch.sum(
            responsibility * compression_variance, dim=-1
        )
        selection_term = temperature * torch.sum(
            responsibility
            * (
                torch.log(
                    responsibility.clamp_min(self.config.eps)
                )
                + math.log(self.K)
            ),
            dim=-1,
        )
        return mismatch_term, variance_term, selection_term

    def coordinate_vicreg(
        self,
        adjacency: torch.Tensor,
        bases: torch.Tensor,
        transport: torch.Tensor,
        measure: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """VICReg anti-collapse regularizer on graph-to-base responses.

        The coupling is detached before recomputing GW energy. Therefore the
        response loss updates bases but cannot be minimized by changing AOT
        correspondences. Returns variance loss, covariance loss, responses,
        mean coordinate standard deviation, and off-diagonal covariance RMS.
        """
        response_energy = self.aot_gw_energy(
            adjacency,
            bases,
            transport.detach(),
            measure.detach(),
        )
        responses = torch.sqrt(
            response_energy.clamp_min(0.0) + self.config.coord_eps
        )
        centered = responses - responses.mean(dim=0, keepdim=True)
        coordinate_std = torch.sqrt(
            centered.square().mean(dim=0) + self.config.coord_eps
        )
        variance_loss = F.relu(
            self.config.coord_std_target - coordinate_std
        ).square().mean()

        batch_size, num_bases = responses.shape
        covariance_loss = responses.new_zeros(())
        offdiag_rms = responses.new_zeros(())
        if batch_size > 1 and num_bases > 1:
            covariance = centered.transpose(0, 1) @ centered
            covariance = covariance / float(batch_size - 1)
            offdiag_mask = ~torch.eye(
                num_bases,
                device=responses.device,
                dtype=torch.bool,
            )
            offdiag = covariance[offdiag_mask]
            covariance_loss = offdiag.square().mean()
            offdiag_rms = torch.sqrt(covariance_loss)
        return (
            variance_loss,
            covariance_loss,
            responses,
            coordinate_std.mean(),
            offdiag_rms,
        )

    def coordinate_vicreg_sparse(
        self,
        graph: SparseGraphBatch,
        bases: torch.Tensor,
        transport: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sparse equivalent of :meth:`coordinate_vicreg`."""
        response_energy = self.aot_gw_energy_sparse(
            graph,
            bases,
            transport.detach(),
        )
        responses = torch.sqrt(
            response_energy.clamp_min(0.0) + self.config.coord_eps
        )
        centered = responses - responses.mean(dim=0, keepdim=True)
        coordinate_std = torch.sqrt(
            centered.square().mean(dim=0) + self.config.coord_eps
        )
        variance_loss = F.relu(
            self.config.coord_std_target - coordinate_std
        ).square().mean()
        batch_size, num_bases = responses.shape
        covariance_loss = responses.new_zeros(())
        offdiag_rms = responses.new_zeros(())
        if batch_size > 1 and num_bases > 1:
            covariance = centered.transpose(0, 1) @ centered
            covariance = covariance / float(batch_size - 1)
            offdiag_mask = ~torch.eye(
                num_bases,
                device=responses.device,
                dtype=torch.bool,
            )
            offdiag = covariance[offdiag_mask]
            covariance_loss = offdiag.square().mean()
            offdiag_rms = torch.sqrt(covariance_loss)
        return (
            variance_loss,
            covariance_loss,
            responses,
            coordinate_std.mean(),
            offdiag_rms,
        )

    def coverage_volume_regularization(
        self,
        adjacency: torch.Tensor,
        bases: torch.Tensor,
        transport: torch.Tensor,
        measure: torch.Tensor,
        energies: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Coverage distortion minus response-space log-det volume.

        Current soft energy responsibilities are treated as a fixed
        assignment for coverage. The volume response recomputes energy with
        detached transports so only the atlas, not AOT correspondence, can
        increase coordinate volume.
        """
        cfg = self.config
        with torch.autocast(
            device_type=adjacency.device.type, enabled=False
        ):
            energies_fp32 = energies.float()
            assignment = F.softmax(
                -energies_fp32.detach() / cfg.mean_temperature,
                dim=-1,
            )
            coverage = torch.sum(
                assignment * energies_fp32, dim=-1
            ).mean()

            response_energy = self.aot_gw_energy(
                adjacency.float(),
                bases.float(),
                transport.detach().float(),
                measure.detach().float(),
            )
            responses = torch.sqrt(
                response_energy.clamp_min(0.0)
                + cfg.coordinate_volume_eps
            )
            centered = responses - responses.mean(dim=0, keepdim=True)
            divisor = max(int(responses.shape[0]) - 1, 1)
            covariance = centered.transpose(0, 1) @ centered
            covariance = covariance / float(divisor)
            covariance = 0.5 * (
                covariance + covariance.transpose(0, 1)
            )
            eye = torch.eye(
                self.K,
                device=responses.device,
                dtype=torch.float32,
            )
            regularized_covariance = (
                covariance + cfg.coordinate_volume_eps * eye
            )
            sign, logabsdet = torch.linalg.slogdet(
                regularized_covariance
            )
            coordinate_volume = logabsdet
            volume_loss = -coordinate_volume
            auxiliary = (
                coverage
                + cfg.coordinate_volume_weight * volume_loss
            )

            eigenvalues = torch.linalg.eigvalsh(
                covariance.detach()
            ).clamp_min(0.0)
            eigenvalue_sum = eigenvalues.sum()
            normalized = eigenvalues / eigenvalue_sum.clamp_min(cfg.eps)
            effective_rank = torch.exp(
                -torch.sum(
                    normalized
                    * torch.log(normalized.clamp_min(cfg.eps))
                )
            )
            effective_rank = torch.where(
                eigenvalue_sum > cfg.eps,
                effective_rank,
                effective_rank.new_zeros(()),
            )
            assignment_entropy = -torch.sum(
                assignment
                * torch.log(assignment.clamp_min(cfg.eps)),
                dim=-1,
            ).mean()
        return {
            "coverage": coverage,
            "coordinate_volume": coordinate_volume,
            "volume_loss": volume_loss,
            "auxiliary": auxiliary,
            "responses": responses,
            "covariance": covariance,
            "effective_rank": effective_rank,
            "assignment_entropy": assignment_entropy,
            "logdet_sign": sign,
            "eigenvalue_min": eigenvalues.min(),
            "eigenvalue_max": eigenvalues.max(),
        }

    def coverage_volume_regularization_sparse(
        self,
        graph: SparseGraphBatch,
        bases: torch.Tensor,
        transport: torch.Tensor,
        energies: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Sparse equivalent of response-space coverage--volume."""
        cfg = self.config
        with torch.autocast(
            device_type=transport.device.type,
            enabled=False,
        ):
            energies_fp32 = energies.float()
            assignment = F.softmax(
                -energies_fp32.detach() / cfg.mean_temperature,
                dim=-1,
            )
            coverage = torch.sum(
                assignment * energies_fp32,
                dim=-1,
            ).mean()
            response_energy = self.aot_gw_energy_sparse(
                graph,
                bases.float(),
                transport.detach().float(),
            )
            responses = torch.sqrt(
                response_energy.clamp_min(0.0)
                + cfg.coordinate_volume_eps
            )
            centered = responses - responses.mean(dim=0, keepdim=True)
            divisor = max(int(responses.shape[0]) - 1, 1)
            covariance = centered.transpose(0, 1) @ centered
            covariance = covariance / float(divisor)
            covariance = 0.5 * (
                covariance + covariance.transpose(0, 1)
            )
            eye = torch.eye(
                self.K,
                device=responses.device,
                dtype=torch.float32,
            )
            regularized_covariance = (
                covariance + cfg.coordinate_volume_eps * eye
            )
            sign, logabsdet = torch.linalg.slogdet(
                regularized_covariance
            )
            coordinate_volume = logabsdet
            volume_loss = -coordinate_volume
            auxiliary = (
                coverage
                + cfg.coordinate_volume_weight * volume_loss
            )
            eigenvalues = torch.linalg.eigvalsh(
                covariance.detach()
            ).clamp_min(0.0)
            eigenvalue_sum = eigenvalues.sum()
            normalized = eigenvalues / eigenvalue_sum.clamp_min(cfg.eps)
            effective_rank = torch.exp(
                -torch.sum(
                    normalized
                    * torch.log(normalized.clamp_min(cfg.eps))
                )
            )
            effective_rank = torch.where(
                eigenvalue_sum > cfg.eps,
                effective_rank,
                effective_rank.new_zeros(()),
            )
            assignment_entropy = -torch.sum(
                assignment
                * torch.log(assignment.clamp_min(cfg.eps)),
                dim=-1,
            ).mean()
        return {
            "coverage": coverage,
            "coordinate_volume": coordinate_volume,
            "volume_loss": volume_loss,
            "auxiliary": auxiliary,
            "responses": responses,
            "covariance": covariance,
            "effective_rank": effective_rank,
            "assignment_entropy": assignment_entropy,
            "logdet_sign": sign,
            "eigenvalue_min": eigenvalues.min(),
            "eigenvalue_max": eigenvalues.max(),
        }

    def coordinate_effective_rank_regularization(
        self,
        energies: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Penalize response effective rank below the configured target.

        The response coordinates are centered first within every graph and
        then across the batch. This removes graph-wide energy offsets and
        measures whether relative graph-to-base responses span enough
        independent directions. Unlike the atlas-only VICReg variants, the
        input energies are not detached, so this loss can shape both the AOT
        correspondence network and learned bases.
        """
        cfg = self.config
        with torch.autocast(
            device_type=energies.device.type,
            enabled=False,
        ):
            energies_fp32 = energies.float()
            target_value = max(
                1.0,
                float(self.K)
                * float(cfg.coordinate_rank_target_fraction),
            )
            target = energies_fp32.new_tensor(target_value)
            target_log = torch.log(target.clamp_min(1.0))
            zero = energies_fp32.sum() * 0.0
            one = zero + 1.0
            required_batch_size = math.ceil(target_value) + 1
            batch_size = int(energies_fp32.shape[0])
            rank_upper_bound = energies_fp32.new_tensor(
                float(max(0, min(batch_size - 1, self.K - 1)))
            )
            batch_valid = energies_fp32.new_tensor(
                float(batch_size >= required_batch_size and self.K >= 2)
            )
            if energies_fp32.shape[0] < 2 or self.K < 2:
                return {
                    "loss": zero,
                    "raw_loss": zero,
                    "effective_rank": one.detach(),
                    "target_rank": target,
                    "spectral_entropy": zero.detach(),
                    "spectral_mass": zero.detach(),
                    "relative_std": zero.detach(),
                    "rank_upper_bound": rank_upper_bound,
                    "batch_valid": batch_valid,
                }

            responses = torch.sqrt(
                energies_fp32.clamp_min(0.0) + cfg.coord_eps
            )
            relative = responses - responses.mean(dim=1, keepdim=True)
            centered = relative - relative.mean(dim=0, keepdim=True)
            covariance = centered.transpose(0, 1) @ centered
            covariance = covariance / float(responses.shape[0] - 1)
            covariance = 0.5 * (
                covariance + covariance.transpose(0, 1)
            )
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            spectral_mass = eigenvalues.sum()
            probabilities = eigenvalues / spectral_mass.clamp_min(cfg.eps)
            spectral_entropy = -torch.sum(
                probabilities
                * torch.log(probabilities.clamp_min(cfg.eps))
            )
            has_spectrum = spectral_mass > cfg.eps
            spectral_entropy = torch.where(
                has_spectrum,
                spectral_entropy,
                zero,
            )
            effective_rank = torch.where(
                has_spectrum,
                torch.exp(spectral_entropy),
                one,
            )
            raw_loss = F.relu(target_log - spectral_entropy).square()
            # A batch with B samples can have rank at most B - 1 after
            # centering. Skip the auxiliary update when the target is unreachable
            # instead of optimizing an impossible per-batch target.
            loss = raw_loss * batch_valid
            relative_std = torch.sqrt(
                covariance.diagonal().mean().clamp_min(0.0)
            )
        return {
            "loss": loss,
            "raw_loss": raw_loss,
            "effective_rank": effective_rank,
            "target_rank": target,
            "spectral_entropy": spectral_entropy,
            "spectral_mass": spectral_mass,
            "relative_std": relative_std,
            "rank_upper_bound": rank_upper_bound,
            "batch_valid": batch_valid,
        }

    def forward(
        self, batch_data
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        graph = self.prepare_sparse_batch(batch_data)
        bases = self.get_normalized_bases()
        base_context = self.encode_base_nodes(bases)
        transport, _measure, residual, _graph_context = (
            self.predict_transport_sparse_batch(
                graph,
                bases=bases,
                base_context=base_context,
            )
        )
        compact_variance = None
        if self.config.loss_variant == "variance":
            compact_variance, energies = (
                self.compact_conditional_variance_sparse(
                    graph,
                    bases,
                    transport,
                )
            )
        else:
            energies = self.aot_gw_energy_sparse(
                graph,
                bases,
                transport,
            )
        mean_losses = self.mean_coverage(energies)
        aot_weights = F.softmax(
            -energies / self.config.mean_temperature, dim=-1
        )
        entropies = -torch.sum(
            aot_weights
            * torch.log(aot_weights.clamp_min(self.config.eps)),
            dim=-1,
        )

        loss_mean = mean_losses.mean()
        logs = {
            "loss_mean": loss_mean.detach(),
            "aot_energy_mean": energies.mean().detach(),
            "aot_entropy": entropies.mean().detach(),
            "sinkhorn_residual": residual.max().detach(),
        }
        if (
            self.training
            and self.config.base_training == "ema_centroid"
        ):
            ema_stage_logs = self.stage_ema_base_update_sparse(
                graph,
                transport,
                energies,
            )
            logs.update(
                {
                    name: value.detach()
                    for name, value in ema_stage_logs.items()
                }
            )
        if self.config.loss_variant == "mean":
            total = self.config.mean_loss_weight * loss_mean
        elif self.config.loss_variant == "decomposed":
            mismatch, compression_variance = (
                self.aot_gw_energy_decomposition_sparse(
                    graph,
                    bases,
                    transport,
                )
            )
            push_term, variance_term, selection_term = (
                self.mean_coverage_decomposition(
                    energies,
                    mismatch,
                    compression_variance,
                )
            )
            loss_pushforward = push_term.mean()
            loss_variance = variance_term.mean()
            loss_selection = selection_term.mean()
            weighted_pushforward = (
                self.config.pushforward_weight * loss_pushforward
            )
            weighted_variance = (
                self.config.variance_weight * loss_variance
            )
            weighted_selection = (
                self.config.selection_weight * loss_selection
            )
            total = (
                weighted_pushforward
                + weighted_variance
                + weighted_selection
            )
            logs.update(
                {
                    "loss_pushforward": loss_pushforward.detach(),
                    "loss_variance": loss_variance.detach(),
                    "loss_selection": loss_selection.detach(),
                    "weighted_pushforward": (
                        weighted_pushforward.detach()
                    ),
                    "weighted_variance": weighted_variance.detach(),
                    "weighted_selection": weighted_selection.detach(),
                }
            )
        else:
            if compact_variance is None:
                raise RuntimeError("Compact variance was not computed.")
            detached_routing = aot_weights.detach()
            loss_variance = torch.sum(
                detached_routing * compact_variance,
                dim=-1,
            ).mean()
            total = loss_variance
            logs.update(
                {
                    "loss_variance": loss_variance.detach(),
                    "variance_mean": compact_variance.mean().detach(),
                }
            )
        if self.config.coordinate_regularization == "vicreg":
            (
                coord_variance,
                coord_covariance,
                coord_responses,
                coord_std_mean,
                coord_cov_offdiag_rms,
            ) = self.coordinate_vicreg_sparse(
                graph,
                bases,
                transport,
            )
            weighted_coord_variance = (
                self.config.coord_variance_weight * coord_variance
            )
            weighted_coord_covariance = (
                self.config.coord_covariance_weight * coord_covariance
            )
            total = (
                total
                + weighted_coord_variance
                + weighted_coord_covariance
            )
            logs.update(
                {
                    "loss_coord_variance": coord_variance.detach(),
                    "loss_coord_covariance": coord_covariance.detach(),
                    "weighted_coord_variance": (
                        weighted_coord_variance.detach()
                    ),
                    "weighted_coord_covariance": (
                        weighted_coord_covariance.detach()
                    ),
                    "coord_response_mean": (
                        coord_responses.mean().detach()
                    ),
                    "coord_std_mean": coord_std_mean.detach(),
                    "coord_cov_offdiag_rms": (
                        coord_cov_offdiag_rms.detach()
                    ),
                }
            )
        elif self.config.coordinate_regularization == "coverage_volume":
            coverage_volume = self.coverage_volume_regularization_sparse(
                graph,
                bases,
                transport,
                energies,
            )
            weighted_auxiliary = (
                self.config.coverage_volume_aux_weight
                * coverage_volume["auxiliary"]
            )
            total = total + weighted_auxiliary
            logs.update(
                {
                    "loss_coverage": (
                        coverage_volume["coverage"].detach()
                    ),
                    "coordinate_volume_logdet": (
                        coverage_volume["coordinate_volume"].detach()
                    ),
                    "loss_coordinate_volume": (
                        coverage_volume["volume_loss"].detach()
                    ),
                    "loss_coverage_volume_aux": (
                        coverage_volume["auxiliary"].detach()
                    ),
                    "weighted_coverage_volume_aux": (
                        weighted_auxiliary.detach()
                    ),
                    "coordinate_effective_rank": (
                        coverage_volume["effective_rank"].detach()
                    ),
                    "coordinate_assignment_entropy": (
                        coverage_volume["assignment_entropy"].detach()
                    ),
                    "coordinate_logdet_sign": (
                        coverage_volume["logdet_sign"].detach()
                    ),
                    "coordinate_eigenvalue_min": (
                        coverage_volume["eigenvalue_min"].detach()
                    ),
                    "coordinate_eigenvalue_max": (
                        coverage_volume["eigenvalue_max"].detach()
                    ),
                }
            )
        elif self.config.coordinate_regularization == "effective_rank":
            rank_result = self.coordinate_effective_rank_regularization(
                energies
            )
            weighted_rank = (
                self.config.coordinate_rank_weight * rank_result["loss"]
            )
            total = total + weighted_rank
            logs.update(
                {
                    "loss_coordinate_effective_rank": (
                        rank_result["loss"].detach()
                    ),
                    "weighted_coordinate_effective_rank": (
                        weighted_rank.detach()
                    ),
                    "coordinate_effective_rank": (
                        rank_result["effective_rank"].detach()
                    ),
                    "coordinate_effective_rank_target": (
                        rank_result["target_rank"].detach()
                    ),
                    "coordinate_spectral_entropy": (
                        rank_result["spectral_entropy"].detach()
                    ),
                    "coordinate_spectral_mass": (
                        rank_result["spectral_mass"].detach()
                    ),
                    "coordinate_relative_std": (
                        rank_result["relative_std"].detach()
                    ),
                    "coordinate_rank_upper_bound": (
                        rank_result["rank_upper_bound"].detach()
                    ),
                    "coordinate_rank_batch_valid": (
                        rank_result["batch_valid"].detach()
                    ),
                    "loss_coordinate_effective_rank_raw": (
                        rank_result["raw_loss"].detach()
                    ),
                }
            )
        if self.config.base_regularization == "separation":
            separation = self.base_separation_regularization(bases)
            weighted_separation = (
                self.config.base_separation_weight
                * separation["loss"]
            )
            total = total + weighted_separation
            logs.update(
                {
                    "loss_base_separation": separation["loss"].detach(),
                    "weighted_base_separation": (
                        weighted_separation.detach()
                    ),
                    "base_pairwise_distance_min": (
                        separation["distance_min"]
                    ),
                    "base_pairwise_distance_mean": (
                        separation["distance_mean"]
                    ),
                    "base_pairwise_distance_median": (
                        separation["distance_median"]
                    ),
                    "base_separation_active_pair_ratio": (
                        separation["active_pair_ratio"]
                    ),
                }
            )
        logs["total"] = total.detach()
        return total, logs



