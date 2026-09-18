from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scgfm_art.node_data import (
    ALL_NODE_DATASETS,
    EXTERNAL_NODE_TARGETS,
    preprocess_node_dataset,
)


def synthetic_graph(seed: int, feature_dim: int) -> Data:
    generator = torch.Generator().manual_seed(seed)
    num_nodes = 32
    source = torch.arange(num_nodes)
    target = (source + 1) % num_nodes
    chords = (source + 5) % num_nodes
    edge_index = torch.stack(
        [
            torch.cat([source, target, source, chords]),
            torch.cat([target, source, chords, source]),
        ]
    )
    labels = torch.arange(num_nodes) % 2
    features = torch.randn(
        num_nodes, feature_dim, generator=generator
    )
    return Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        num_nodes=num_nodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.output_dir)
    for index, name in enumerate(ALL_NODE_DATASETS):
        preprocess_node_dataset(
            synthetic_graph(index + 1, 4 + index),
            name,
            root,
            max_subgraph_nodes=16,
            shard_size=16,
            ppr_workers=args.num_threads,
            center_scope=(
                "per_class_cap"
                if name in EXTERNAL_NODE_TARGETS
                else "all_nodes"
            ),
            samples_per_class=(
                16 if name in EXTERNAL_NODE_TARGETS else None
            ),
            force=True,
        )
    print(f"Prepared synthetic SCGFM-ART PPR smoke data: {root}")


if __name__ == "__main__":
    main()

