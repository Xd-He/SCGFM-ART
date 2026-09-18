# SCGFM-ART

Official implementation of **SCGFM-ART: Amortized Relational Transport for
Structure-Centric Graph Foundation Models**.

SCGFM-ART maps heterogeneous graphs to a shared relational atlas. A sparse GIN
encodes normalized node degrees, an amortized transport network predicts
graph-to-base couplings, and mean relational coverage trains the atlas without
labels. Frozen downstream representations concatenate global ART energy
coordinates with transported node features:

$$
z(G) = [q(G)\|\mathrm{vec}(H(G))].
$$

This release contains only the SCGFM-ART model and the two cross-domain
classification protocols reported in Tables II and III. It does not include
third-party baselines, ablations, pretrained weights, datasets, or result
files.

## Supported protocols

| Task | Source domains | Additional unseen targets | Main head |
|---|---|---|---|
| Graph classification | NCI1, BZR, COLLAB, IMDB-BINARY, PROTEINS | COLORS-3, ogbg-molhiv | 5-shot ProtoNet |
| Node classification | Cora, CiteSeer, PubMed, Computers, Photo | Reddit, ogbn-arxiv | 5-shot linear probe |

The five source domains use leave-one-dataset-out (LODO) pretraining.
Additional targets use a checkpoint pretrained jointly on all five sources.
Both protocols use 50 deterministic episodes by default.

## Installation

Python 3.10 or newer is required. The code was validated with Python 3.10,
PyTorch 2.9.1+cu126, PyG 2.7.0, NumPy 2.2.6, scikit-learn 1.7.2,
OGB 1.3.6, and Numba 0.65.1.

Install the PyTorch build matching your CUDA driver first, following the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).
Then install this project:

```bash
conda activate GFM_env
cd SCGFM-ART
python -m pip install -e ".[test]"
```

CPU execution is supported for smoke tests. Full experiments are intended for
a CUDA GPU; BF16 autocast is enabled by default on CUDA.

## Data layout

All downloaded or generated data stays under the ignored `data/` directory:

```text
data/
├── TUDataset/          # PyG TU datasets and ogbg-molhiv cache
└── node_ppr_k400/      # prepared node-centric PPR subgraphs
```

Graph datasets are downloaded by PyG/OGB when first requested. Node datasets
must be converted once to sharded PPR subgraphs. Prepare the five pretraining
sources with all center nodes:

```bash
python scripts/prepare_node_data.py \
  --raw-data-root data \
  --output-dir data/node_ppr_k400 \
  --datasets Cora,CiteSeer,PubMed,Computers,Photo \
  --max-subgraph-nodes 400 \
  --ppr-alpha 0.15 \
  --ppr-eps 1e-5 \
  --shard-size 512 \
  --center-scope all_nodes \
  --num-threads 28 \
  --seed 42 \
  --resume
```

Prepare target-only Reddit and ogbn-arxiv with at most 300 centers per class:

```bash
python scripts/prepare_node_data.py \
  --raw-data-root data \
  --output-dir data/node_ppr_k400 \
  --datasets Reddit,ogbn-arxiv \
  --max-subgraph-nodes 400 \
  --ppr-alpha 0.15 \
  --ppr-eps 1e-5 \
  --shard-size 512 \
  --center-scope per_class_cap \
  --samples-per-class 300 \
  --num-threads 28 \
  --seed 42 \
  --resume
```

## Reproduce Table II: graph classification

```bash
python scripts/run_graph_classification.py \
  --data-root data/TUDataset \
  --output-dir outputs/graph \
  --device cuda \
  --epochs 60 \
  --batch-size 64 \
  --encoder-batch-size 64 \
  --num-workers 4 \
  --encoder-num-workers 4 \
  --amp-dtype bf16 \
  --k-shot 5 \
  --n-query 50 \
  --n-runs 50 \
  --resume
```

The reported metric is mean ProtoNet accuracy over 50 episodes.

## Reproduce Table III: node classification

```bash
python scripts/run_node_classification.py \
  --prepared-data-root data/node_ppr_k400 \
  --output-dir outputs/node \
  --device cuda \
  --epochs 60 \
  --batch-size 64 \
  --encoder-batch-size 64 \
  --num-workers 4 \
  --encoder-num-workers 4 \
  --amp-dtype bf16 \
  --samples-per-class 300 \
  --random-projection-dim 256 \
  --random-projection-seed 42 \
  --k-shot 5 \
  --n-query 50 \
  --n-runs 50 \
  --linear-epochs 1000 \
  --linear-lr 1.0 \
  --resume
```

The reported metric is mean linear-probe accuracy over 50 episodes. Gaussian
RP256 is deterministic for seed 42 and is applied only during downstream
encoding.

## Command-line arguments

All three programs support `--help`, for example:

```bash
python scripts/run_graph_classification.py --help
python scripts/prepare_node_data.py --help
python scripts/run_node_classification.py --help
```

### Shared model and training arguments

These options are available in both classification programs.

| Argument | Default | Description |
|---|---:|---|
| `--device` | `auto` | Execution device. Use `cuda`, `cuda:0`, or `cpu`; `auto` selects CUDA when available. |
| `--seed` | `42` | Random seed for model initialization, sampling, and few-shot episodes. |
| `--K` | `16` | Number of learnable relational bases in the atlas. |
| `--M` | `32` | Number of relational roles in each base. |
| `--hidden-dim` | `64` | Hidden width of the normalized-degree GIN and ART predictor. |
| `--num-gin-layers` | `2` | Number of GIN message-passing layers. |
| `--sinkhorn-iterations` | `20` | Number of Sinkhorn projections used to enforce transport marginals. |
| `--sinkhorn-temperature` | `0.1` | Temperature used when predicting graph-to-base couplings. |
| `--mean-temperature` | `0.1` | Soft-min temperature for mean relational coverage and base responsibilities. |
| `--top-k` | `8` | Number of highest-responsibility bases mixed for feature recoding. |
| `--epochs` | `60` | Number of unsupervised pretraining epochs per LODO fold. |
| `--batch-size` | `64` | Pretraining graph/subgraph batch size. |
| `--lr` | `1e-3` | Adam learning rate for model pretraining. |
| `--grad-clip` | `5.0` | Maximum gradient norm applied before each optimizer step. |
| `--num-workers` | `4` | Number of pretraining DataLoader workers. Set to `0` when debugging worker issues. |
| `--prefetch-factor` | `2` | Batches prefetched by each DataLoader worker; relevant only when workers are enabled. |
| `--encoder-batch-size` | `64` | Batch size used to generate frozen downstream embeddings. |
| `--encoder-num-workers` | `4` | Number of DataLoader workers used during frozen encoding. |
| `--amp-dtype` | `bf16` | Encoder autocast type: `bf16`, `float16`, or `float32`. Sinkhorn and relational energy remain FP32. |
| `--no-amp` | off | Disable mixed precision. Useful for CPU execution or numerical diagnosis. |
| `--heatmap-interval` | `20` | Save an atlas heatmap every this many epochs. |
| `--resume` | off | Continue incomplete training and reuse compatible completed checkpoints/evaluations. |
| `--fail-fast` | off | Stop at the first failed fold instead of recording it and continuing. |
| `--smoke` | off | Override expensive settings with a tiny CPU-friendly validation run. |

Changing $K$, $M$, hidden width, Sinkhorn settings, or the training seed changes
the checkpoint signature. Use a new output directory unless the run is an
exact continuation.

### Graph-classification arguments

| Argument | Default | Description |
|---|---|---|
| `--sources` | five paper sources | Comma-separated LODO source datasets: `NCI1,BZR,COLLAB,IMDB-BINARY,PROTEINS`. |
| `--external-targets` | two paper targets | Comma-separated targets evaluated with the all-source checkpoint: `COLORS-3,ogbg-molhiv`. |
| `--skip-external` | off | Run only source-domain LODO folds and skip the two external targets. |
| `--data-root` | `data/TUDataset` | TU dataset directory. The OGB loader derives its cache parent from this path. |
| `--output-dir` | `outputs/graph` | Checkpoints, embeddings, episode rows, and aggregate accuracy tables. |
| `--max-nodes` | `1000` | Ignore graphs larger than this number of nodes. |
| `--max-per-class` | unlimited | Optional deterministic per-class cap, primarily intended for debugging. |
| `--k-shot` | `5` | Number of labeled support graphs per class in each episode. |
| `--n-query` | `50` | Maximum number of query graphs per class in each episode. |
| `--n-runs` | `50` | Number of deterministic ProtoNet episodes. |

For a target source dataset $D$, its fold is pretrained on all names in
`--sources` except $D$. The target never enters that fold's pretraining set.

### Node-data preprocessing arguments

| Argument | Default | Description |
|---|---:|---|
| `--raw-data-root` | `data` | Parent directory used by PyG and OGB to download or locate raw node datasets. |
| `--output-dir` | `data/node_ppr_k400` | Destination for manifests, feature tensors, labels, centers, and subgraph shards. |
| `--datasets` | all seven datasets | Comma-separated dataset names to prepare. Separate source and external runs because they use different center scopes. |
| `--max-subgraph-nodes` | `400` | Maximum nodes retained in each PPR subgraph. |
| `--ppr-alpha` | `0.15` | Personalized PageRank restart probability. |
| `--ppr-eps` | `1e-4` | Local-push residual threshold. The paper commands explicitly use `1e-5`. |
| `--shard-size` | `250` | Number of prepared subgraphs stored in each shard. The paper commands use `512`. |
| `--center-scope` | `all_nodes` | `all_nodes` for pretraining sources; `per_class_cap` for target-only datasets. |
| `--samples-per-class` | `300` | Center-node limit per class when `--center-scope per_class_cap` is selected. |
| `--max-pushes` | `1000000` | Safety limit on local-push operations for one center node. |
| `--num-threads` | `28` | CPU threads/processes used by PPR preprocessing. Reduce this on smaller machines. |
| `--seed` | `42` | Seed for deterministic center selection and preprocessing. |
| `--resume` | off | Reuse verified completed shards and continue an interrupted preprocessing run. |
| `--force` | off | Rebuild the requested prepared dataset even if output already exists. Do not combine with `--resume`. |

Prepared source datasets must use `all_nodes`; Reddit and ogbn-arxiv use
`per_class_cap` because they are target-only domains in the paper protocol.

### Node-classification arguments

| Argument | Default | Description |
|---|---:|---|
| `--prepared-data-root` | `data/node_ppr_k400` | Root containing the prepared dataset manifests and shards. |
| `--sources` | five paper sources | `Cora,CiteSeer,PubMed,Computers,Photo`; each selected target is held out from pretraining. |
| `--targets` | all sources | Optional comma-separated subset of source LODO targets, useful for distributing folds across GPUs. |
| `--external-targets` | two paper targets | `Reddit,ogbn-arxiv`, evaluated with the checkpoint pretrained on all sources. |
| `--skip-external` | off | Skip Reddit and ogbn-arxiv evaluation. |
| `--output-dir` | `outputs/node` | Checkpoints, fixed episode splits, encoded unions, and accuracy summaries. |
| `--samples-per-class` | `300` | Deterministic target evaluation cap per class; source pretraining still uses every prepared source center. Use `0` for no cap. |
| `--random-projection-dim` | `256` | Width of the Gaussian projection applied to target node attributes. |
| `--random-projection-seed` | `42` | Seed used to construct the deterministic projection matrix. |
| `--k-shot` | `5` | Number of labeled support nodes per class in each episode. |
| `--n-query` | `50` | Maximum number of query nodes per class in each episode. |
| `--n-runs` | `50` | Number of deterministic evaluation episodes. |
| `--linear-epochs` | `1000` | Maximum optimization iterations for the downstream linear classifier. |
| `--linear-lr` | `1.0` | Learning rate used by the released linear-probe protocol. |

## Defaults and outputs

The paper configuration uses $K=16$ bases, $M=32$ roles per base, hidden
dimension 64, two GIN layers, top-8 coupling mixing, 20 Sinkhorn iterations,
learning rate $10^{-3}$, and 60 pretraining epochs.

Each LODO fold writes its checkpoint, training history, configuration,
deterministic episode splits, per-episode metrics, and summary below the chosen
`--output-dir`. Root-level CSV and JSON files aggregate all targets.
`--resume` reuses compatible completed checkpoints and continues incomplete
training.

If GPU memory is insufficient, reduce `--batch-size` and
`--encoder-batch-size` first. Keep $K$, $M$, the PPR size, and RP256 fixed
when reproducing the paper.

## Smoke tests

Graph smoke test using two locally cached TU datasets:

```bash
python scripts/run_graph_classification.py \
  --data-root data/TUDataset \
  --sources BZR,PROTEINS \
  --skip-external \
  --device cpu \
  --output-dir outputs/smoke_graph \
  --smoke
```

Build synthetic node data and run a two-domain node smoke test:

```bash
python tests/build_node_smoke_data.py --output-dir data/smoke_node
python scripts/run_node_classification.py \
  --prepared-data-root data/smoke_node \
  --sources Cora,CiteSeer \
  --skip-external \
  --device cpu \
  --output-dir outputs/smoke_node \
  --smoke
```

Run the unit suite with:

```bash
python -m pytest
```

## License and citation

The code is released under the MIT License. If you have any questions, contact by email: hexiaodong24@126.com.
