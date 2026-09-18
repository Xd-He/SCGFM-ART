from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj

from scgfm_art.checkpoint import load_checkpoint, save_checkpoint
from scgfm_art.encoder import (
    SCGFMARTEncoder,
    gaussian_random_projection_matrix,
)
from scgfm_art.fewshot import (
    create_episode_splits,
    episode_index_union,
    evaluate_fewshot,
    evaluate_fewshot_splits,
)
from scgfm_art.model import SCGFMARTConfig, SCGFMARTModel
from scgfm_art.node_data import (
    PreparedNodeSubgraphDataset,
    preprocess_node_dataset,
)
from scripts.run_graph_classification import lodo_training_sources


def graph(label: int = 0, feature_dim: int = 3) -> Data:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    x = torch.arange(4 * feature_dim, dtype=torch.float32).reshape(
        4, feature_dim
    )
    return Data(edge_index=edge_index, x=x, y=torch.tensor([label]), num_nodes=4)


def model() -> SCGFMARTModel:
    torch.manual_seed(7)
    return SCGFMARTModel(
        SCGFMARTConfig(
            K=3,
            M=4,
            hidden_dim=12,
            num_gin_layers=2,
            sinkhorn_iterations=20,
        )
    )


def test_atlas_is_symmetric_with_zero_diagonal():
    bases = model().get_normalized_bases()
    assert torch.allclose(bases, bases.transpose(-1, -2), atol=1e-6)
    assert torch.allclose(
        torch.diagonal(bases, dim1=-2, dim2=-1),
        torch.zeros(3, 4),
    )


def test_normalized_degree_is_the_only_gin_input():
    current = model()
    adjacency = torch.zeros(4, 4)
    adjacency[0, 1] = adjacency[1, 0] = 1
    adjacency[1, 2] = adjacency[2, 1] = 1
    captured: list[torch.Tensor] = []
    handle = current.graph_input.register_forward_pre_hook(
        lambda _module, values: captured.append(values[0].detach().clone())
    )
    try:
        current.encode_graph_nodes(adjacency)
    finally:
        handle.remove()
    assert current.graph_input.weight.shape == (12, 1)
    assert torch.equal(
        captured[0], torch.tensor([[0.5], [1.0], [0.5], [0.0]])
    )


def test_sparse_energy_and_gradients_match_dense_oracle():
    batch = Batch.from_data_list([graph(), graph(1)])
    dense_model = model()
    sparse_model = model()
    sparse_model.load_state_dict(dense_model.state_dict())

    sparse_graph = dense_model.prepare_sparse_batch(batch)
    dense = to_dense_adj(
        sparse_graph.edge_index,
        sparse_graph.batch,
        max_num_nodes=sparse_graph.max_num_nodes,
    ).float()
    bases = dense_model.get_normalized_bases()
    base_context = dense_model.encode_base_nodes(bases)
    transport, measure, _ = dense_model.predict_transport_batch(
        dense,
        sparse_graph.node_mask,
        bases=bases,
        base_context=base_context,
    )
    dense_energy = dense_model.aot_gw_energy(
        dense, bases, transport, measure
    )
    dense_model.mean_coverage(dense_energy).mean().backward()

    sparse_graph_2 = sparse_model.prepare_sparse_batch(batch)
    sparse_bases = sparse_model.get_normalized_bases()
    sparse_context = sparse_model.encode_base_nodes(sparse_bases)
    sparse_transport, _, _, _ = sparse_model.predict_transport_sparse_batch(
        sparse_graph_2,
        bases=sparse_bases,
        base_context=sparse_context,
    )
    sparse_energy = sparse_model.aot_gw_energy_sparse(
        sparse_graph_2, sparse_bases, sparse_transport
    )
    sparse_model.mean_coverage(sparse_energy).mean().backward()

    assert torch.allclose(sparse_energy, dense_energy, atol=1e-6)
    for name, parameter in dense_model.named_parameters():
        other = dict(sparse_model.named_parameters())[name]
        assert parameter.grad is not None and other.grad is not None
        assert torch.allclose(parameter.grad, other.grad, atol=2e-6, rtol=2e-5)


def test_transport_marginals_and_mean_coverage_are_finite():
    current = model()
    batch = Batch.from_data_list([graph(), graph(1)])
    sparse = current.prepare_sparse_batch(batch)
    transports, measure, residual, _ = current.predict_transport_sparse_batch(
        sparse
    )
    assert transports.shape == (2, 3, 4, 4)
    assert torch.allclose(transports.sum(-1), measure[:, None, :], atol=1e-5)
    expected_columns = torch.full((2, 3, 4), 0.25)
    assert torch.allclose(transports.sum(-2), expected_columns, atol=1e-5)
    assert float(residual.detach().max()) < 1e-4
    energies = current.aot_gw_energy_sparse(sparse, current.get_normalized_bases(), transports)
    assert torch.isfinite(current.mean_coverage(energies)).all()


def test_art_full_representation_dimension():
    encoder = SCGFMARTEncoder(model(), device="cpu", top_k=2)
    embedding = encoder.encode_batch(Batch.from_data_list([graph()]))
    assert embedding.shape == (1, 3 + 4 * 3)
    assert torch.isfinite(embedding).all()


def test_checkpoint_round_trip_uses_release_version():
    current = model()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        save_checkpoint(current, path, epoch=4, train_config={"epochs": 6})
        restored, payload = load_checkpoint(path, "cpu")
        assert payload["variant"] == "scgfm_art_release_v1"
        assert payload["epoch"] == 4
        for name, value in current.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name])


def test_ppr_preprocessing_and_sharded_loading():
    data = graph()
    data.y = torch.tensor([0, 0, 1, 1])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        preprocess_node_dataset(
            data,
            "Cora",
            root,
            max_subgraph_nodes=4,
            shard_size=2,
            ppr_workers=1,
            center_scope="all_nodes",
        )
        prepared = PreparedNodeSubgraphDataset(
            root, "Cora", with_features=True, verify_checksums=True
        )
        assert len(prepared) == 4
        assert prepared.centers.tolist() == [0, 1, 2, 3]
        assert prepared[0].x.shape[1] == 3
        assert prepared[0].center_node_id == 0


def test_random_projection_is_deterministic():
    first = gaussian_random_projection_matrix(7, 256, seed=42)
    second = gaussian_random_projection_matrix(7, 256, seed=42)
    different = gaussian_random_projection_matrix(7, 256, seed=43)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_graph_protocol_reports_only_prototype_accuracy():
    embeddings = torch.cat(
        [torch.randn(12, 4) - 3, torch.randn(12, 4) + 3]
    )
    labels = torch.tensor([0] * 12 + [1] * 12)
    summary, rows = evaluate_fewshot(
        embeddings,
        labels,
        k_shot=2,
        n_query=4,
        n_runs=3,
        seed=42,
        device="cpu",
    )
    assert set(summary["heads"]) == {"prototype"}
    assert {row["head"] for row in rows} == {"prototype"}


def test_node_protocol_reports_only_linear_accuracy_when_requested():
    embeddings = torch.cat(
        [torch.randn(12, 4) - 3, torch.randn(12, 4) + 3]
    )
    labels = torch.tensor([0] * 12 + [1] * 12)
    splits = create_episode_splits(
        labels, k_shot=2, n_query=4, n_runs=2, seed=42
    )
    union = episode_index_union(splits)
    row_map = torch.full((labels.numel(),), -1, dtype=torch.long)
    row_map[union] = torch.arange(union.numel())
    summary, rows = evaluate_fewshot_splits(
        embeddings[union],
        labels,
        row_map,
        splits,
        device="cpu",
        linear_epochs=50,
        linear_lr=0.5,
        heads=("linear",),
    )
    assert set(summary["heads"]) == {"linear"}
    assert {row["head"] for row in rows} == {"linear"}


def test_lodo_excludes_target_and_rejects_unknown_target():
    sources = ["NCI1", "BZR", "COLLAB"]
    selected = lodo_training_sources(sources, "BZR")
    assert selected == ["NCI1", "COLLAB"]
    assert "BZR" not in selected
    try:
        lodo_training_sources(sources, "MUTAG")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown target must be rejected.")
