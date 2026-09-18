from __future__ import annotations

import bisect
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import warnings
from collections import OrderedDict
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid, Reddit
from torch_geometric.utils import (
    coalesce,
    to_undirected,
)


NODE_PREPROCESS_VERSION = "scgfm_art_ppr_all_centers_v1"
NODE_TARGET_PREPROCESS_VERSION = (
    "scgfm_art_ppr_target_classcap_v1"
)
SUPPORTED_NODE_PREPROCESS_VERSIONS = frozenset(
    {NODE_PREPROCESS_VERSION, NODE_TARGET_PREPROCESS_VERSION}
)
SOURCE_NODE_DATASETS = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo"]
EXTERNAL_NODE_TARGETS = ["Reddit", "ogbn-arxiv"]
ALL_NODE_DATASETS = SOURCE_NODE_DATASETS + EXTERNAL_NODE_TARGETS


def canonical_node_dataset_name(name: str) -> str:
    aliases = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "computers": "Computers",
        "computer": "Computers",
        "amazon-computers": "Computers",
        "amazon-compuerts": "Computers",
        "photo": "Photo",
        "amazon-photo": "Photo",
        "reddit": "Reddit",
        "ogbn-arxiv": "ogbn-arxiv",
    }
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError(
            f"Unsupported node dataset {name!r}; expected one of "
            f"{ALL_NODE_DATASETS}."
        )
    return aliases[key]


def dataset_directory(root: str | Path, name: str) -> Path:
    canonical = canonical_node_dataset_name(name)
    root = Path(root)
    normalized = canonical.lower().replace("_", "-")
    candidates = [
        root / canonical.lower(),
        root / canonical,
        root / canonical.replace("-", "_"),
    ]
    unique_candidates = list(dict.fromkeys(candidates))
    for candidate in unique_candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    if root.is_dir():
        for child in root.iterdir():
            child_key = child.name.lower().replace("_", "-")
            if (
                child.is_dir()
                and child_key == normalized
                and (child / "manifest.json").is_file()
            ):
                return child
    # New preprocessing keeps the established lowercase layout.
    return unique_candidates[0]


def _has_pyg_dataset_payload(path: Path) -> bool:
    for folder_name in ("processed", "raw"):
        folder = path / folder_name
        if folder.is_dir() and any(item.is_file() for item in folder.iterdir()):
            return True
    return False


def raw_dataset_root(root: str | Path, name: str) -> Path:
    """Resolve the server's direct dataset layout with legacy fallback."""
    root = Path(root)
    canonical = canonical_node_dataset_name(name)
    if canonical == "ogbn-arxiv":
        # OGB appends ``ogbn_arxiv`` to the supplied parent directory.
        if root.name.lower().replace("-", "_") == "ogbn_arxiv":
            if _has_pyg_dataset_payload(root / "ogbn_arxiv"):
                return root
            return root.parent
        direct = root / "ogbn_arxiv"
        if _has_pyg_dataset_payload(direct / "ogbn_arxiv"):
            return direct
        legacy_parent = root / "OGB"
        if direct.exists() or not (legacy_parent / "ogbn_arxiv").exists():
            return root
        return legacy_parent

    # Also accept a dataset-specific root such as ``--raw-data-root
    # ../data/Cora`` without incorrectly resolving it to ``Cora/Cora``.
    if root.name.casefold() == canonical.casefold():
        nested = root / canonical
        if (
            not _has_pyg_dataset_payload(root)
            and _has_pyg_dataset_payload(nested)
        ):
            return nested
        return root
    direct = root / canonical
    nested = direct / canonical
    if (
        not _has_pyg_dataset_payload(direct)
        and _has_pyg_dataset_payload(nested)
    ):
        return nested
    if direct.exists():
        return direct
    if canonical in {"Cora", "CiteSeer", "PubMed"}:
        legacy = root / "Planetoid"
    elif canonical in {"Computers", "Photo"}:
        legacy = root / "Amazon"
    else:
        legacy = direct
    return legacy if legacy.exists() else direct


def _load_single_graph_processed_file(root: Path) -> Data | None:
    """Load PyG's single-graph cache without triggering its download hook."""
    path = root / "processed" / "data.pt"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Data):
        return payload
    if not isinstance(payload, tuple) or len(payload) not in (2, 3):
        raise ValueError(f"Unsupported PyG processed payload: {path}")
    stored = payload[0]
    data_class = payload[2] if len(payload) == 3 else Data
    if isinstance(stored, dict):
        stored = data_class.from_dict(stored)
    if not isinstance(stored, Data):
        raise TypeError(
            f"Expected a homogeneous PyG Data object in {path}, "
            f"found {type(stored).__name__}."
        )
    return stored


def load_raw_node_dataset(root: str | Path, name: str) -> Data:
    root = Path(root)
    canonical = canonical_node_dataset_name(name)
    resolved_root = raw_dataset_root(root, canonical)
    data = _load_single_graph_processed_file(resolved_root)
    if data is not None:
        pass
    elif canonical in {"Cora", "CiteSeer", "PubMed"}:
        data = Planetoid(root=str(resolved_root), name=canonical)[0]
    elif canonical in {"Computers", "Photo"}:
        data = Amazon(root=str(resolved_root), name=canonical)[0]
    elif canonical == "Reddit":
        data = Reddit(root=str(resolved_root))[0]
    else:
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            from torch_geometric.data.data import (
                DataEdgeAttr,
                DataTensorAttr,
            )
            from torch_geometric.data.storage import GlobalStorage

            torch.serialization.add_safe_globals(
                [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]
            )
        except ImportError as exc:
            raise RuntimeError(
                "ogbn-arxiv preprocessing requires the `ogb` package."
            ) from exc
        data = PygNodePropPredDataset(
            name="ogbn-arxiv", root=str(resolved_root)
        )[0]
    data.y = data.y.view(-1).long()
    return data


def canonicalize_node_graph(data: Data) -> Data:
    num_nodes = int(data.num_nodes)
    # Match SCGFM-ART's formal baseline preprocessing: make the graph
    # undirected and coalesce duplicates, but do not remove self-loops.
    edge_index = to_undirected(
        data.edge_index.long(), num_nodes=num_nodes
    )
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    labels = data.y.view(-1).long()
    features = (
        data.x.detach().cpu().float().contiguous()
        if getattr(data, "x", None) is not None
        else torch.zeros((num_nodes, 1), dtype=torch.float32)
    )
    return Data(
        x=features,
        edge_index=edge_index.cpu().contiguous(),
        y=labels.cpu().contiguous(),
        num_nodes=num_nodes,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_descriptor(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _csr(edge_index: torch.Tensor, num_nodes: int):
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    order = np.lexsort((col, row))
    row = row[order]
    col = col[order]
    counts = np.bincount(row, minlength=num_nodes)
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    return rowptr, col.astype(np.int64, copy=False)


class SCGFMOriginalLocalPushPPR:
    """Exact local-push implementation used by SCGFM-ART."""

    def __init__(
        self,
        rowptr: np.ndarray,
        columns: np.ndarray,
        *,
        alpha: float = 0.15,
        epsilon: float = 1.0e-4,
        max_pushes: int = 1_000_000,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if max_pushes < 1:
            raise ValueError("max_pushes must be positive.")
        self.rowptr = np.asarray(rowptr, dtype=np.int64)
        self.columns = np.asarray(columns, dtype=np.int64)
        self.num_nodes = self.rowptr.size - 1
        self.degrees = np.diff(self.rowptr).astype(
            np.float32, copy=False
        )
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.max_pushes = int(max_pushes)

    def push(self, seed: int) -> tuple[np.ndarray, int, bool]:
        """Return the local-push score ``reserve + residual``."""
        reserve = np.zeros(self.num_nodes, dtype=np.float32)
        residual = np.zeros(self.num_nodes, dtype=np.float32)
        in_queue = np.zeros(self.num_nodes, dtype=np.bool_)
        residual[int(seed)] = 1.0
        queue: deque[int] = deque([int(seed)])
        in_queue[int(seed)] = True
        pushes = 0

        while queue and pushes < self.max_pushes:
            node = queue.popleft()
            in_queue[node] = False
            residual_mass = float(residual[node])
            degree = float(self.degrees[node])
            threshold = self.epsilon * max(degree, 1.0)
            if residual_mass <= threshold and not (
                pushes == 0 and node == int(seed)
            ):
                continue

            residual[node] = (
                (1.0 - self.alpha) * residual_mass / 2.0
            )
            reserve[node] += self.alpha * residual_mass
            pushes += 1

            if residual[node] > threshold and not in_queue[node]:
                queue.append(node)
                in_queue[node] = True
            if degree == 0.0:
                continue

            left = self.rowptr[node]
            right = self.rowptr[node + 1]
            neighbors = self.columns[left:right]
            share = (
                (1.0 - self.alpha) * residual_mass / (2.0 * degree)
            )
            np.add.at(residual, neighbors, share)
            neighbor_degrees = self.degrees[neighbors]
            active = neighbors[
                residual[neighbors]
                > self.epsilon
                * np.maximum(neighbor_degrees, 1.0)
            ]
            for candidate in np.unique(active):
                candidate = int(candidate)
                if not in_queue[candidate]:
                    queue.append(candidate)
                    in_queue[candidate] = True

        return reserve + residual, pushes, bool(queue)

    def select(
        self,
        seed: int,
        maximum: int,
    ) -> tuple[np.ndarray, np.ndarray, int, bool]:
        scores, pushes, truncated = self.push(seed)
        count = min(int(maximum), scores.size)
        # Deliberately preserve the release protocol behavior: when fewer than
        # ``maximum`` nodes have positive PPR, zero-score nodes still fill
        # the remaining positions.
        selected = np.argpartition(scores, -count)[-count:]
        selected = selected[np.argsort(scores[selected])[::-1]]
        return (
            selected.astype(np.int64, copy=False),
            scores[selected].astype(np.float32, copy=False),
            pushes,
            truncated,
        )


_PPR_PROCESS_WORKER: SCGFMOriginalLocalPushPPR | None = None
_PPR_PROCESS_MAXIMUM = 0


def _initialize_ppr_process(
    rowptr: np.ndarray,
    columns: np.ndarray,
    alpha: float,
    epsilon: float,
    max_pushes: int,
    maximum: int,
) -> None:
    global _PPR_PROCESS_WORKER, _PPR_PROCESS_MAXIMUM
    _PPR_PROCESS_WORKER = SCGFMOriginalLocalPushPPR(
        rowptr,
        columns,
        alpha=alpha,
        epsilon=epsilon,
        max_pushes=max_pushes,
    )
    _PPR_PROCESS_MAXIMUM = int(maximum)


def _select_ppr_process(
    center: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    if _PPR_PROCESS_WORKER is None:
        raise RuntimeError("PPR process worker was not initialized.")
    return _PPR_PROCESS_WORKER.select(
        int(center), _PPR_PROCESS_MAXIMUM
    )


def _induced_local_edges(
    selected: np.ndarray,
    rowptr: np.ndarray,
    columns: np.ndarray,
    marker: np.ndarray,
) -> torch.Tensor:
    marker[selected] = np.arange(selected.size, dtype=np.int64)
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for local_source, global_source in enumerate(selected):
        neighbors = columns[
            rowptr[global_source] : rowptr[global_source + 1]
        ]
        mapped = marker[neighbors]
        keep = mapped >= 0
        if np.any(keep):
            target = mapped[keep]
            source_parts.append(
                np.full(target.size, local_source, dtype=np.int64)
            )
            target_parts.append(target.astype(np.int64, copy=False))
    marker[selected] = -1
    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(
        np.stack(
            [np.concatenate(source_parts), np.concatenate(target_parts)]
        )
    ).long()


def _build_shard(
    data: Data,
    centers: torch.Tensor,
    rowptr: np.ndarray,
    columns: np.ndarray,
    max_subgraph_nodes: int,
    ppr_alpha: float,
    ppr_eps: float,
    max_pushes: int,
    executor: ProcessPoolExecutor | None,
) -> dict[str, torch.Tensor]:
    node_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []
    edge_chunks: list[torch.Tensor] = []
    center_local: list[int] = []
    node_ptr = [0]
    edge_ptr = [0]
    marker = np.full(int(data.num_nodes), -1, dtype=np.int64)
    centers_list = [int(value) for value in centers.tolist()]
    if executor is None:
        ppr = SCGFMOriginalLocalPushPPR(
            rowptr,
            columns,
            alpha=ppr_alpha,
            epsilon=ppr_eps,
            max_pushes=max_pushes,
        )
        selections = [
            ppr.select(center, max_subgraph_nodes)
            for center in centers_list
        ]
    else:
        selections = list(
            executor.map(_select_ppr_process, centers_list, chunksize=1)
        )

    total_pushes = 0
    truncated_count = 0
    for center, selection in zip(centers_list, selections):
        selected, scores, pushes, truncated = selection
        total_pushes += int(pushes)
        truncated_count += int(truncated)
        edges = _induced_local_edges(
            selected, rowptr, columns, marker
        )
        node_tensor = torch.from_numpy(selected).long()
        node_chunks.append(node_tensor)
        score_chunks.append(torch.from_numpy(scores).float())
        edge_chunks.append(edges)
        local_positions = np.flatnonzero(selected == center)
        if local_positions.size != 1:
            raise RuntimeError(
                f"Center node {center} was not selected exactly once."
            )
        center_local.append(int(local_positions[0]))
        node_ptr.append(node_ptr[-1] + selected.size)
        edge_ptr.append(edge_ptr[-1] + edges.shape[1])

    return {
        "node_ids": torch.cat(node_chunks),
        "ppr_scores": torch.cat(score_chunks),
        "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
        "edge_index": torch.cat(edge_chunks, dim=1),
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),
        "center_node_ids": centers.cpu().long(),
        "center_local_indices": torch.tensor(center_local, dtype=torch.long),
        "labels": data.y[centers].cpu().long(),
        "total_pushes": torch.tensor(total_pushes, dtype=torch.long),
        "truncated_count": torch.tensor(
            truncated_count, dtype=torch.long
        ),
    }


def preprocess_node_dataset(
    data: Data,
    dataset_name: str,
    output_root: str | Path,
    *,
    max_subgraph_nodes: int = 400,
    ppr_alpha: float = 0.15,
    ppr_eps: float = 1e-4,
    shard_size: int = 512,
    max_pushes: int = 1_000_000,
    ppr_workers: int = 1,
    seed: int = 42,
    center_scope: str = "all_nodes",
    samples_per_class: int | None = None,
    resume: bool = False,
    force: bool = False,
) -> Path:
    if max_subgraph_nodes < 1:
        raise ValueError("max_subgraph_nodes must be positive.")
    if shard_size < 1:
        raise ValueError("shard_size must be positive.")
    if ppr_workers < 1:
        raise ValueError("ppr_workers must be positive.")
    if center_scope not in {"all_nodes", "per_class_cap"}:
        raise ValueError(
            "center_scope must be 'all_nodes' or 'per_class_cap'."
        )
    if center_scope == "all_nodes" and samples_per_class is not None:
        warnings.warn(
            "samples_per_class is a legacy preprocessing argument and "
            "is ignored. Every node will be used exactly once as a "
            "center; class capping is applied only during downstream "
            "evaluation.",
            stacklevel=2,
        )
    if center_scope == "per_class_cap" and (
        samples_per_class is None or samples_per_class < 1
    ):
        raise ValueError(
            "per_class_cap preprocessing requires a positive "
            "samples_per_class."
        )
    canonical = canonical_node_dataset_name(dataset_name)
    output = dataset_directory(output_root, canonical).resolve()
    root = Path(output_root).resolve()
    if root not in output.parents:
        raise ValueError(f"Unsafe preprocessing output path: {output}")
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)

    data = canonicalize_node_graph(data)
    if center_scope == "all_nodes":
        centers = torch.arange(int(data.num_nodes), dtype=torch.long)
        version = NODE_PREPROCESS_VERSION
    else:
        rng = np.random.default_rng(seed)
        labels_numpy = data.y.cpu().numpy()
        selected: list[int] = []
        for label in np.unique(labels_numpy):
            candidates = np.flatnonzero(labels_numpy == label)
            if candidates.size > int(samples_per_class):
                candidates = rng.choice(
                    candidates,
                    int(samples_per_class),
                    replace=False,
                )
            selected.extend(int(index) for index in candidates)
        rng.shuffle(selected)
        centers = torch.tensor(selected, dtype=torch.long)
        version = NODE_TARGET_PREPROCESS_VERSION
    config = {
        "version": version,
        "dataset": canonical,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "num_features": int(data.x.shape[1]),
        "num_classes": int(torch.unique(data.y).numel()),
        "max_subgraph_nodes": int(max_subgraph_nodes),
        "ppr_alpha": float(ppr_alpha),
        "ppr_eps": float(ppr_eps),
        "ppr_method": "scgfm_acl_local_push_exact",
        "max_pushes": int(max_pushes),
        "shard_size": int(shard_size),
        "center_scope": center_scope,
    }
    if center_scope == "per_class_cap":
        config["samples_per_class"] = int(samples_per_class)
        config["sampling_seed"] = int(seed)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in config}
        if comparable != config:
            raise ValueError(
                f"Preprocessing config mismatch at {manifest_path}: "
                f"saved={comparable}, requested={config}. Use --force or "
                "a new output directory."
            )
        if not resume and existing.get("complete"):
            raise FileExistsError(
                f"Prepared dataset already exists at {output}; use --resume."
            )
        manifest = existing
    else:
        manifest = {
            **config,
            "complete": False,
            "class_counts": {
                str(int(label)): int(count)
                for label, count in zip(
                    *torch.unique(data.y, return_counts=True)
                )
            },
            "shards": [],
        }

    features_path = output / "features.pt"
    labels_path = output / "labels.pt"
    centers_path = output / "centers.pt"
    if not features_path.exists():
        _atomic_torch_save(data.x.cpu(), features_path)
    if not labels_path.exists():
        _atomic_torch_save(data.y.cpu(), labels_path)
    if not centers_path.exists():
        _atomic_torch_save(centers, centers_path)
    artifacts = {
        "features": _artifact_descriptor(features_path, output),
        "labels": _artifact_descriptor(labels_path, output),
        "centers": _artifact_descriptor(centers_path, output),
    }
    saved_artifacts = manifest.get("artifacts")
    if saved_artifacts is not None and saved_artifacts != artifacts:
        raise ValueError(
            f"Prepared artifact checksum mismatch at {output}. "
            "Use --force to rebuild the dataset."
        )
    manifest["artifacts"] = artifacts

    rowptr, columns = _csr(data.edge_index, int(data.num_nodes))
    total_centers = int(centers.numel())
    total_shards = math.ceil(total_centers / shard_size)
    known = {int(item["index"]): item for item in manifest["shards"]}
    executor = None
    if ppr_workers > 1:
        start_method = (
            "fork"
            if "fork" in mp.get_all_start_methods()
            else "spawn"
        )
        executor = ProcessPoolExecutor(
            max_workers=ppr_workers,
            mp_context=mp.get_context(start_method),
            initializer=_initialize_ppr_process,
            initargs=(
                rowptr,
                columns,
                ppr_alpha,
                ppr_eps,
                max_pushes,
                max_subgraph_nodes,
            ),
        )
    try:
        for shard_index in range(total_shards):
            start = shard_index * shard_size
            end = min(total_centers, start + shard_size)
            relative = Path("shards") / f"part_{shard_index:06d}.pt"
            shard_path = output / relative
            saved = known.get(shard_index)
            if (
                resume
                and saved is not None
                and shard_path.exists()
                and _sha256(shard_path) == saved.get("sha256")
            ):
                continue
            shard_centers = centers[start:end]
            payload = _build_shard(
                data,
                shard_centers,
                rowptr,
                columns,
                max_subgraph_nodes,
                ppr_alpha,
                ppr_eps,
                max_pushes,
                executor,
            )
            _atomic_torch_save(payload, shard_path)
            descriptor = {
                "index": shard_index,
                "start": start,
                "end": end,
                "count": end - start,
                "path": relative.as_posix(),
                "sha256": _sha256(shard_path),
                "max_nodes": int(
                    (
                        payload["node_ptr"][1:]
                        - payload["node_ptr"][:-1]
                    )
                    .max()
                    .item()
                ),
                "total_pushes": int(payload["total_pushes"]),
                "truncated_count": int(payload["truncated_count"]),
            }
            known[shard_index] = descriptor
            manifest["shards"] = [
                known[index] for index in sorted(known)
            ]
            manifest["complete"] = False
            _atomic_json(manifest, manifest_path)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    manifest["shards"] = [known[index] for index in range(total_shards)]
    manifest["num_subgraphs"] = int(
        sum(item["count"] for item in manifest["shards"])
    )
    manifest["total_pushes"] = int(
        sum(item.get("total_pushes", 0) for item in manifest["shards"])
    )
    manifest["truncated_centers"] = int(
        sum(
            item.get("truncated_count", 0)
            for item in manifest["shards"]
        )
    )
    manifest["complete"] = manifest["num_subgraphs"] == total_centers
    if not manifest["complete"]:
        raise RuntimeError(f"Incomplete preprocessing at {output}.")
    _atomic_json(manifest, manifest_path)
    return output


class PreparedNodeSubgraphDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        name: str,
        *,
        with_features: bool,
        cache_size: int = 2,
        verify_checksums: bool = False,
    ) -> None:
        self.directory = dataset_directory(root, name)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            root_path = Path(root).expanduser()
            visible = (
                sorted(item.name for item in root_path.iterdir())
                if root_path.is_dir()
                else []
            )
            raise FileNotFoundError(
                "Prepared node-data manifest was not found. "
                f"dataset={canonical_node_dataset_name(name)!r}, "
                f"prepared_root={root_path.resolve()}, "
                f"expected={manifest_path.resolve()}, "
                f"visible_entries={visible}."
            )
        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            self.manifest.get("version")
            not in SUPPORTED_NODE_PREPROCESS_VERSIONS
            or not self.manifest.get("complete")
        ):
            raise ValueError(
                f"Invalid or incomplete prepared dataset: {self.directory}"
            )
        self.name = self.manifest["dataset"]
        self.with_features = bool(with_features)
        self.cache_size = max(1, int(cache_size))
        self.shards = self.manifest["shards"]
        self.ends = [int(item["end"]) for item in self.shards]
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self.features = (
            torch.load(
                self.directory / "features.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if with_features
            else None
        )
        self.full_labels = torch.load(
            self.directory / "labels.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        ).long()
        self.centers = torch.load(
            self.directory / "centers.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        ).long()
        self.labels = self.full_labels[self.centers]
        if self.centers.numel() != len(self):
            raise ValueError(
                "Prepared center count does not match num_subgraphs."
            )
        center_scope = self.manifest.get("center_scope")
        if center_scope == "all_nodes":
            expected_centers = torch.arange(
                int(self.manifest["num_nodes"]), dtype=torch.long
            )
            if not torch.equal(self.centers, expected_centers):
                raise ValueError(
                    "An all-node prepared dataset must contain every node "
                    "exactly once in ascending global-node-ID order."
                )
        elif center_scope == "per_class_cap":
            if (
                self.centers.unique().numel() != self.centers.numel()
                or self.centers.numel() == 0
                or int(self.centers.min()) < 0
                or int(self.centers.max()) >= int(self.manifest["num_nodes"])
            ):
                raise ValueError(
                    "Invalid center IDs in per-class-capped target cache."
                )
        else:
            raise ValueError(
                f"Unsupported prepared center_scope: {center_scope!r}"
            )
        if verify_checksums:
            for item in self.manifest.get("artifacts", {}).values():
                path = self.directory / item["path"]
                if (
                    not path.exists()
                    or path.stat().st_size != int(item["size_bytes"])
                    or _sha256(path) != item["sha256"]
                ):
                    raise ValueError(
                        f"Prepared artifact checksum mismatch: {path}"
                    )
            for item in self.shards:
                path = self.directory / item["path"]
                if not path.exists() or _sha256(path) != item["sha256"]:
                    raise ValueError(f"Shard checksum mismatch: {path}")

    def __len__(self) -> int:
        return int(self.manifest["num_subgraphs"])

    @property
    def shard_ranges(self) -> list[tuple[int, int]]:
        return [
            (int(item["start"]), int(item["end"])) for item in self.shards
        ]

    def _load_shard(self, index: int) -> dict:
        if index in self._cache:
            payload = self._cache.pop(index)
            self._cache[index] = payload
            return payload
        payload = torch.load(
            self.directory / self.shards[index]["path"],
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self._cache[index] = payload
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> Data:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        shard = self._load_shard(shard_index)
        local = index - int(self.shards[shard_index]["start"])
        node_left = int(shard["node_ptr"][local])
        node_right = int(shard["node_ptr"][local + 1])
        edge_left = int(shard["edge_ptr"][local])
        edge_right = int(shard["edge_ptr"][local + 1])
        node_ids = shard["node_ids"][node_left:node_right].clone()
        graph = Data(
            x=(
                None
                if self.features is None
                else self.features[node_ids].clone()
            ),
            edge_index=shard["edge_index"][
                :, edge_left:edge_right
            ].clone(),
            y=shard["labels"][local].view(1).clone(),
            num_nodes=node_right - node_left,
        )
        graph.dataset_name = self.name
        graph.source_index = int(index)
        graph.center_node_id = int(shard["center_node_ids"][local])
        graph.center_local_index = int(
            shard["center_local_indices"][local]
        )
        graph.global_node_ids = node_ids
        return graph


class ShardAwareSampler(Sampler[int]):
    """Shuffle shard blocks and their members while covering every item once."""

    def __init__(self, dataset: Dataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.blocks: list[list[int]] = []
        if isinstance(dataset, ConcatDataset):
            offset = 0
            for child in dataset.datasets:
                ranges = getattr(child, "shard_ranges", [(0, len(child))])
                self.blocks.extend(
                    [list(range(offset + left, offset + right)) for left, right in ranges]
                )
                offset += len(child)
        else:
            ranges = getattr(dataset, "shard_ranges", [(0, len(dataset))])
            self.blocks = [list(range(left, right)) for left, right in ranges]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.blocks)))
        rng.shuffle(order)
        for block_index in order:
            block = list(self.blocks[block_index])
            rng.shuffle(block)
            yield from block


def prepared_subset(dataset: Dataset, indices: Sequence[int]) -> Dataset:
    from torch.utils.data import Subset

    return Subset(dataset, [int(index) for index in indices])



