from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset


SOURCE_DATASETS = ["NCI1", "BZR", "COLLAB", "IMDB-BINARY", "PROTEINS"]
EXTERNAL_TARGETS = ["COLORS-3", "ogbg-molhiv"]


def resolve_tu_root(data_root: str | Path, name: str) -> Path:
    root = Path(data_root)
    if (root / name).exists():
        return root
    if (root / "TUDataset" / name).exists():
        return root / "TUDataset"
    return root


def load_graphs(
    data_root: str | Path,
    name: str,
    max_nodes: int | None = None,
    max_per_class: int | None = None,
    seed: int = 42,
    drop_node_features: bool = False,
):
    root = Path(data_root)
    if name.lower().startswith("ogbg-"):
        try:
            from ogb.graphproppred import PygGraphPropPredDataset
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
                "OGB datasets require the `ogb` package in GFM_env."
            ) from exc
        ogb_root = root.parent if root.name.lower() == "tudataset" else root
        dataset = PygGraphPropPredDataset(
            name=name, root=str(ogb_root)
        )
    else:
        dataset = TUDataset(
            root=str(resolve_tu_root(root, name)),
            name=name,
            use_node_attr=True,
        )
    graphs = []
    for source_index, raw in enumerate(dataset):
        if (
            getattr(raw, "y", None) is None
            or raw.y.numel() != 1
            or not torch.isfinite(raw.y.float()).all()
        ):
            continue
        num_nodes = (
            int(raw.num_nodes)
            if raw.num_nodes is not None
            else int(raw.edge_index.max().item()) + 1
        )
        if max_nodes is not None and num_nodes > max_nodes:
            continue
        graph = Data(
            edge_index=raw.edge_index.long(),
            x=(
                None
                if drop_node_features or getattr(raw, "x", None) is None
                else raw.x
            ),
            y=raw.y.view(1).long(),
            num_nodes=num_nodes,
        )
        graph.dataset_name = name
        graph.source_index = int(source_index)
        graph.domain_id = 0
        graphs.append(graph)
    if max_per_class is not None:
        groups = defaultdict(list)
        for graph in graphs:
            groups[int(graph.y.view(-1)[0].item())].append(graph)
        rng = random.Random(seed)
        selected = []
        for label in sorted(groups):
            rng.shuffle(groups[label])
            selected.extend(groups[label][:max_per_class])
        graphs = selected
    if not graphs:
        raise RuntimeError(f"No usable graphs found for {name}.")
    return graphs


def set_domain_id(graphs: Sequence[Data], domain_id: int) -> list[Data]:
    for graph in graphs:
        graph.domain_id = int(domain_id)
    return list(graphs)


def structure_only_copies(graphs: Sequence[Data]) -> list[Data]:
    copies = []
    for graph in graphs:
        structural = Data(
            edge_index=graph.edge_index,
            x=None,
            y=graph.y,
            num_nodes=int(graph.num_nodes),
        )
        structural.dataset_name = getattr(graph, "dataset_name", "")
        structural.source_index = int(
            getattr(graph, "source_index", len(copies))
        )
        structural.domain_id = int(getattr(graph, "domain_id", 0))
        copies.append(structural)
    return copies


class DomainBalancedSampler(Sampler[int]):
    """Uniform-domain sampling with a fixed number of graphs per epoch."""

    def __init__(
        self,
        graphs: Sequence[Data],
        seed: int = 42,
    ) -> None:
        self.graphs = graphs
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[int, list[int]] = defaultdict(list)
        for index, graph in enumerate(graphs):
            self.groups[int(getattr(graph, "domain_id", 0))].append(index)
        if not self.groups:
            raise ValueError("DomainBalancedSampler requires graphs.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.graphs)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        pools = {domain: list(values) for domain, values in self.groups.items()}
        for values in pools.values():
            rng.shuffle(values)
        positions = {domain: 0 for domain in pools}
        domains = sorted(pools)
        produced = 0
        while produced < len(self.graphs):
            cycle = list(domains)
            rng.shuffle(cycle)
            for domain in cycle:
                if produced >= len(self.graphs):
                    break
                values = pools[domain]
                position = positions[domain]
                if position >= len(values):
                    rng.shuffle(values)
                    position = 0
                yield values[position]
                positions[domain] = position + 1
                produced += 1


