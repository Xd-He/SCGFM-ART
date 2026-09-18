from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scgfm_art.node_data import (
    ALL_NODE_DATASETS,
    canonical_node_dataset_name,
    load_raw_node_dataset,
    preprocess_node_dataset,
    raw_dataset_root,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_raw_data_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()

    def dataset_directory_score(candidate: Path) -> int:
        expected = (
            "Cora",
            "CiteSeer",
            "PubMed",
            "Computers",
            "Photo",
            "Reddit",
            "ogbn_arxiv",
        )
        return sum((candidate / name).is_dir() for name in expected)

    cwd_score = dataset_directory_score(cwd_candidate)
    project_score = dataset_directory_score(project_candidate)
    if project_score > cwd_score:
        return project_candidate
    return cwd_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare node-centric PPR subgraphs for SCGFM-ART."
        )
    )
    parser.add_argument(
        "--raw-data-root", default=str(PROJECT_ROOT.parent / "data")
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT / "data" / "node_ppr_k400"
        ),
    )
    parser.add_argument("--datasets", default=",".join(ALL_NODE_DATASETS))
    parser.add_argument("--max-subgraph-nodes", type=int, default=400)
    parser.add_argument("--ppr-alpha", type=float, default=0.15)
    parser.add_argument("--ppr-eps", type=float, default=1e-4)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument(
        "--center-scope",
        choices=("all_nodes", "per_class_cap"),
        default="all_nodes",
        help=(
            "Use all_nodes for pretraining source domains; use "
            "per_class_cap for target-only domains."
        ),
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=300,
        help="Used only when --center-scope per_class_cap.",
    )
    parser.add_argument("--max-pushes", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=28)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_data_root = resolve_raw_data_root(args.raw_data_root)
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.num_threads < 1:
        raise ValueError("--num-threads must be at least 1.")
    torch.set_num_threads(args.num_threads)
    try:
        torch.set_num_interop_threads(min(4, args.num_threads))
    except RuntimeError:
        pass
    names = [
        canonical_node_dataset_name(item)
        for item in args.datasets.split(",")
        if item.strip()
    ]
    print(
        f"CPU preprocessing: torch_threads={torch.get_num_threads()} "
        f"ppr_processes={args.num_threads}",
        flush=True,
    )
    print(
        f"Raw data parent: {args.raw_data_root}\n"
        f"Prepared output: {args.output_dir}",
        flush=True,
    )
    for name in names:
        resolved = raw_dataset_root(args.raw_data_root, name)
        print(
            f"Load raw node dataset: {name} | root={resolved}",
            flush=True,
        )
        data = load_raw_node_dataset(args.raw_data_root, name)
        output = preprocess_node_dataset(
            data,
            name,
            Path(args.output_dir),
            max_subgraph_nodes=args.max_subgraph_nodes,
            ppr_alpha=args.ppr_alpha,
            ppr_eps=args.ppr_eps,
            shard_size=args.shard_size,
            max_pushes=args.max_pushes,
            ppr_workers=args.num_threads,
            seed=args.seed,
            center_scope=args.center_scope,
            samples_per_class=(
                args.samples_per_class
                if args.center_scope == "per_class_cap"
                else None
            ),
            resume=args.resume,
            force=args.force,
        )
        print(
            f"Prepared SCGFM-ART PPR subgraphs: {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()




