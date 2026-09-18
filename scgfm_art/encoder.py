from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from .model import SCGFMARTModel


@dataclass
class ARTEncodingComponents:
    """Structured output of the frozen ART encoder."""

    energies: torch.Tensor | None
    q: torch.Tensor
    responsibilities: torch.Tensor | None
    transports: torch.Tensor
    mixed_transport: torch.Tensor
    recoded: torch.Tensor
    h_only: torch.Tensor
    aot_full: torch.Tensor


def gaussian_random_projection_matrix(
    input_dim: int,
    output_dim: int = 256,
    seed: int = 42,
) -> torch.Tensor:
    """Return the deterministic Gaussian matrix used by the release protocol."""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Projection dimensions must be positive.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(
        int(input_dim),
        int(output_dim),
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(int(output_dim))


class SCGFMARTEncoder:
    """Frozen SCGFM-ART graph encoder using AOT-predicted transport.

    The main representation is ``[q_AOT || vec(H_AOT)]``. AOT energies
    provide both the explicit coordinates and top-k normalized base mixture.
    Raw target node features (or normalized degree fallback) use the original
    SCGFM node-count recoding ``N T^T X``.
    """

    implementation_version = "scgfm_art_encoder_v1"
    coordinate_modes = ("aot_full",)
    feature_readouts = ("flatten",)
    feature_transport_modes = ("aot_mixture",)
    node_feature_modes = ("raw", "random_projection")

    @classmethod
    def version_for_coordinate_mode(cls, coordinate_mode: str) -> str:
        if coordinate_mode not in cls.coordinate_modes:
            raise ValueError(
                f"Unknown coordinate mode {coordinate_mode!r}; "
                f"expected one of {cls.coordinate_modes}."
            )
        return f"{cls.implementation_version}_{coordinate_mode}"

    @classmethod
    def version_for_configuration(
        cls,
        coordinate_mode: str,
        feature_readout: str,
        feature_transport: str = "aot_mixture",
        node_feature_mode: str = "raw",
        random_projection_dim: int = 256,
        random_projection_seed: int = 42,
    ) -> str:
        base = cls.version_for_coordinate_mode(coordinate_mode)
        if feature_readout not in cls.feature_readouts:
            raise ValueError(
                f"Unknown feature readout {feature_readout!r}; "
                f"expected one of {cls.feature_readouts}."
            )
        if feature_transport not in cls.feature_transport_modes:
            raise ValueError(
                f"Unknown feature transport {feature_transport!r}; "
                f"expected one of {cls.feature_transport_modes}."
            )
        if feature_transport == "mixed_base_aot":
            base = f"{base}_mixed_base_aot"
        if node_feature_mode not in cls.node_feature_modes:
            raise ValueError(
                f"Unknown node feature mode {node_feature_mode!r}; "
                f"expected one of {cls.node_feature_modes}."
            )
        if node_feature_mode == "random_projection":
            if random_projection_dim <= 0:
                raise ValueError("random_projection_dim must be positive.")
            base = (
                f"{base}_rp{int(random_projection_dim)}"
                f"_seed{int(random_projection_seed)}"
            )
        if feature_readout == "pool":
            return f"{base}_pool_max_mean_std"
        if feature_readout == "target":
            return f"{base}_target_barycentric"
        return base

    def __init__(
        self,
        model: SCGFMARTModel,
        tau: float | None = None,
        device: str | torch.device = "cpu",
        max_dim: int = 100,
        num_projections: int = 200,
        top_k: int = 8,
        coordinate_mode: str = "aot_full",
        aot_weight_temperature: float | None = None,
        feature_readout: str = "flatten",
        feature_transport: str = "aot_mixture",
        node_feature_mode: str = "raw",
        random_projection_dim: int = 256,
        random_projection_seed: int = 42,
    ) -> None:
        if coordinate_mode not in self.coordinate_modes:
            raise ValueError(
                f"Unknown coordinate mode {coordinate_mode!r}; "
                f"expected one of {self.coordinate_modes}."
            )
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.tau = model.tau if tau is None else tau
        self.max_dim = max(max_dim, model.M)
        self.num_projections = num_projections
        self.top_k = top_k
        self.coordinate_mode = coordinate_mode
        if feature_readout not in self.feature_readouts:
            raise ValueError(
                f"Unknown feature readout {feature_readout!r}; "
                f"expected one of {self.feature_readouts}."
            )
        self.feature_readout = feature_readout
        if feature_transport not in self.feature_transport_modes:
            raise ValueError(
                f"Unknown feature transport {feature_transport!r}; "
                f"expected one of {self.feature_transport_modes}."
            )
        if (
            feature_transport == "mixed_base_aot"
            and coordinate_mode != "aot_full"
        ):
            raise ValueError(
                "feature_transport='mixed_base_aot' requires "
                "coordinate_mode='aot_full' so that AOT energies define "
                "the base-combination weights."
            )
        self.feature_transport = feature_transport
        if node_feature_mode not in self.node_feature_modes:
            raise ValueError(
                f"Unknown node feature mode {node_feature_mode!r}; "
                f"expected one of {self.node_feature_modes}."
            )
        if random_projection_dim <= 0:
            raise ValueError("random_projection_dim must be positive.")
        self.node_feature_mode = node_feature_mode
        self.random_projection_dim = int(random_projection_dim)
        self.random_projection_seed = int(random_projection_seed)
        self._feature_projection: torch.Tensor | None = None
        self._feature_projection_input_dim: int | None = None
        self.aot_weight_temperature = (
            model.config.mean_temperature
            if aot_weight_temperature is None
            else float(aot_weight_temperature)
        )
        if self.aot_weight_temperature <= 0:
            raise ValueError("aot_weight_temperature must be positive.")
        self.encoder_version = self.version_for_configuration(
            coordinate_mode,
            feature_readout,
            feature_transport,
            node_feature_mode,
            random_projection_dim,
            random_projection_seed,
        )
        with torch.no_grad():
            self.bases = model.get_normalized_bases().detach()

        if coordinate_mode == "aot_full":
            self.theta = None
            self.padded_bases = None
            self.base_projection = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(42)
            self.theta = torch.randn(
                self.max_dim,
                num_projections,
                device=self.device,
                generator=generator,
            )
            self.theta = F.normalize(self.theta, p=2, dim=0)
            self.padded_bases = F.pad(
                self.bases,
                (
                    0,
                    self.max_dim - self.model.M,
                    0,
                    self.max_dim - self.model.M,
                ),
            )
            self.base_projection = torch.sort(
                torch.matmul(self.padded_bases, self.theta), dim=1
            ).values

    @property
    def output_config(self) -> dict:
        return {
            "encoder_version": self.encoder_version,
            "coordinate_mode": self.coordinate_mode,
            "feature_readout": self.feature_readout,
            "feature_transport": self.feature_transport,
            "node_feature_mode": self.node_feature_mode,
            "random_projection_dim": (
                self.random_projection_dim
                if self.node_feature_mode == "random_projection"
                else None
            ),
            "random_projection_seed": (
                self.random_projection_seed
                if self.node_feature_mode == "random_projection"
                else None
            ),
            "random_projection_pre_l2": False,
            "random_projection_post_l2": False,
            "top_k": self.top_k,
            "aot_weight_temperature": self.aot_weight_temperature,
            "representation": (
                "[q_AOT || readout(N T_AOT(A,B_mix)^T X)]"
                if self.feature_transport == "mixed_base_aot"
                else "[q_AOT || readout(N T_AOT^T X)]"
            ),
        }

    def _project_node_features(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Apply SCGFM-ART's deterministic Gaussian RP on the GPU.

        The projection matrix is generated on CPU so that its values exactly
        match ``the release projection protocol`` for the same input
        dimension, output dimension, and seed. Only the batched matrix
        multiplication is performed on the evaluation device.
        """
        if self.node_feature_mode == "raw":
            return features
        input_dim = int(features.shape[-1])
        if (
            self._feature_projection is None
            or self._feature_projection_input_dim != input_dim
        ):
            projection = gaussian_random_projection_matrix(
                input_dim,
                self.random_projection_dim,
                self.random_projection_seed,
            )
            self._feature_projection = projection.to(
                device=self.device, non_blocking=True
            )
            self._feature_projection_input_dim = input_dim
        return torch.matmul(features.float(), self._feature_projection)

    def compute_sliced_gw_distance_fast(
        self, adjacency: torch.Tensor
    ) -> torch.Tensor:
        return self.compute_sliced_gw_distance_batch(adjacency[None])[0]

    def compute_sliced_gw_distance_batch(
        self, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if self.theta is None or self.base_projection is None:
            raise RuntimeError(
                "SGW projection is disabled in coordinate_mode='aot_full'."
            )
        n = adjacency.shape[-1]
        if n < self.max_dim:
            graph = F.pad(
                adjacency,
                (0, self.max_dim - n, 0, self.max_dim - n),
            )
        else:
            graph = adjacency[..., : self.max_dim, : self.max_dim]
        graph_projection = torch.sort(
            torch.matmul(graph, self.theta), dim=1
        ).values
        return torch.mean(
            (
                graph_projection[:, None] - self.base_projection[None]
            ).square(),
            dim=(2, 3),
        )

    @staticmethod
    def feature_matrix(data, adjacency: torch.Tensor) -> torch.Tensor:
        if getattr(data, "x", None) is not None:
            return data.x.to(adjacency.device).float()
        degree = adjacency.sum(dim=1, keepdim=True)
        return degree / degree.max().clamp_min(1.0)

    @torch.no_grad()
    def encode_components_batch(
        self,
        batch: Batch,
    ) -> ARTEncodingComponents:
        batch = batch.to(self.device, non_blocking=True)
        graph = self.model.prepare_sparse_batch(batch)
        counts = graph.counts
        node_mask = graph.node_mask
        if getattr(batch, "x", None) is not None:
            features = self.model.pad_sparse_nodes(
                batch.x.float(),
                graph,
            )
        else:
            maximum = torch.zeros(
                graph.num_graphs,
                device=self.device,
                dtype=graph.degree.dtype,
            )
            if graph.degree.numel() > 0:
                maximum.scatter_reduce_(
                    0,
                    graph.batch,
                    graph.degree,
                    reduce="amax",
                    include_self=True,
                )
            normalized_degree = graph.degree / maximum.index_select(
                0, graph.batch
            ).clamp_min(1.0)
            features = self.model.pad_sparse_nodes(
                normalized_degree[:, None],
                graph,
            )
        features = self._project_node_features(features)
        features = features * node_mask[..., None]

        sgw_weights = None
        if self.coordinate_mode != "aot_full":
            adjacency = to_dense_adj(
                graph.edge_index,
                graph.batch,
                max_num_nodes=graph.max_num_nodes,
            ).float()
            distances = self.compute_sliced_gw_distance_batch(adjacency)
            sgw_weights = F.softmax(-distances / self.tau, dim=-1)
        graph_context = self.model.pad_sparse_nodes(
            self.model.encode_graph_nodes_sparse(graph),
            graph,
        )
        transports, _measure, _, graph_context = (
            self.model.predict_transport_sparse_batch(
                graph,
                bases=self.bases,
                graph_context=graph_context,
            )
        )
        energies = None
        aot_coordinates = None
        aot_weights = None
        if self.coordinate_mode in (
            "aot",
            "sgw_aot",
            "aot_full",
        ):
            energies = self.model.aot_gw_energy_sparse(
                graph,
                self.bases,
                transports,
            )
            aot_coordinates = torch.sqrt(energies.clamp_min(0.0))
            if self.coordinate_mode == "aot_full":
                aot_weights = F.softmax(
                    -energies / self.aot_weight_temperature,
                    dim=-1,
                )
        mixture_weights = (
            aot_weights
            if self.coordinate_mode == "aot_full"
            else sgw_weights
        )
        if mixture_weights is None:
            raise RuntimeError("No base mixture weights were constructed.")
        if self.feature_transport == "mixed_base_aot":
            # Original SCGFM-style composition: use every normalized AOT
            # responsibility to combine one basis, then run the same AOT
            # predictor once more for the graph-to-combined-basis transport.
            mixed_bases = torch.einsum(
                "bk,kmn->bmn", mixture_weights, self.bases
            )
            mixed_transport, _, _ = (
                self.model.predict_transport_to_graph_bases_sparse_batch(
                    graph,
                    mixed_bases,
                    graph_context=graph_context,
                )
            )
        else:
            selected_count = (
                self.model.K
                if self.top_k is None or self.top_k <= 0
                else min(self.top_k, self.model.K)
            )
            selected_weights, selected_indices = torch.topk(
                mixture_weights, k=selected_count, dim=-1
            )
            selected_weights = selected_weights / selected_weights.sum(
                dim=-1, keepdim=True
            )
            sparse_weights = torch.zeros_like(
                mixture_weights
            ).scatter(1, selected_indices, selected_weights)
            mixed_transport = torch.einsum(
                "bk,bknm->bnm", sparse_weights, transports
            )
        recoded = torch.einsum(
            "bnm,bnf->bmf", mixed_transport, features
        )
        # Match the released SCGFM downstream encoder exactly: N * T^T X.
        recoded = recoded * counts[:, None, None]
        if self.coordinate_mode == "sgw":
            coordinates = sgw_weights
        elif self.coordinate_mode in ("aot", "aot_full"):
            coordinates = aot_coordinates
        else:
            if sgw_weights is None or aot_coordinates is None:
                raise RuntimeError("Missing SGW or AOT coordinates.")
            coordinates = (
                torch.cat([sgw_weights, aot_coordinates], dim=-1)
            )
        if self.feature_readout == "flatten":
            feature_embedding = recoded.flatten(start_dim=1)
        elif self.feature_readout == "pool":
            feature_embedding = torch.cat(
                [
                    recoded.amax(dim=1),
                    recoded.mean(dim=1),
                    recoded.std(dim=1, unbiased=False),
                ],
                dim=1,
            )
        else:
            center_local = getattr(batch, "center_local_index", None)
            if center_local is None:
                raise ValueError(
                    "feature_readout='target' requires "
                    "`center_local_index` on every input subgraph."
                )
            center_local = torch.as_tensor(
                center_local, device=self.device
            ).long().view(-1)
            if center_local.numel() != counts.numel():
                raise ValueError(
                    "Expected one center_local_index per input graph."
                )
            # PyG treats every attribute containing ``index`` as an index
            # tensor and offsets it while batching. Restore subgraph-local
            # center positions before indexing the padded transport.
            center_local = center_local - batch.ptr[:-1]
            if torch.any(center_local < 0) or torch.any(
                center_local >= counts
            ):
                raise ValueError(
                    "center_local_index is outside its subgraph."
                )
            graph_ids = torch.arange(
                counts.numel(), device=self.device
            )
            center_transport = mixed_transport[
                graph_ids, center_local
            ]
            conditional = center_transport / center_transport.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
            feature_embedding = torch.einsum(
                "bm,bmf->bf", conditional, recoded
            )
        full = torch.cat([coordinates, feature_embedding], dim=1)
        return ARTEncodingComponents(
            energies=energies,
            q=coordinates,
            responsibilities=aot_weights,
            transports=transports,
            mixed_transport=mixed_transport,
            recoded=recoded,
            h_only=feature_embedding,
            aot_full=full,
        )

    @torch.no_grad()
    def encode_batch(self, batch: Batch) -> torch.Tensor:
        return self.encode_components_batch(batch).aot_full

    @torch.no_grad()
    def encode_single(self, data) -> torch.Tensor:
        return self.encode_batch(Batch.from_data_list([data]))[0]

    @torch.no_grad()
    def encode_dataset(
        self,
        dataset,
        show_progress: bool = True,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        iterator = (
            tqdm(loader, desc="Encode graph batches")
            if show_progress
            else loader
        )
        for batch in iterator:
            labels.append(batch.y.view(-1).long().cpu())
            embeddings.append(self.encode_batch(batch).cpu())
        return torch.cat(embeddings), torch.cat(labels)


