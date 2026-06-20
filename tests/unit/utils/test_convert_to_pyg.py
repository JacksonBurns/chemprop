"""Tests for Chemprop to PyTorch Geometric model conversion.

These tests verify that:
1. Chemprop models can be converted to PyG format via CLI and Python API
2. Converted PyG models produce identical predictions to the original Chemprop models
3. Various configurations (bond/atom message passing, aggregation types, predictors) work
"""

from argparse import Namespace

import pytest
import torch
from torch import Tensor

from chemprop import nn
from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.models.model import MPNN
from chemprop.models.utils import save_model
from chemprop.nn.agg import MeanAggregation, NormAggregation, SumAggregation

torch_geometric = pytest.importorskip("torch_geometric")


@pytest.fixture
def sample_smiles():
    return [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CCN",
        "O=C=O",
        "CC=C",
        "c1ccc2ccccc2c1",
        "OCCN",
    ]


@pytest.fixture
def sample_targets(sample_smiles):
    return torch.rand(len(sample_smiles), 1)


@pytest.fixture
def test_dataset(sample_smiles, sample_targets):
    datapoints = [
        MoleculeDatapoint.from_smi(smi, y.tolist()) for smi, y in zip(sample_smiles, sample_targets)
    ]
    return MoleculeDataset(datapoints)


@pytest.fixture
def test_batch(test_dataset):
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=collate_batch)
    return next(iter(loader))


@pytest.fixture
def sample_molgraph(test_batch):
    return test_batch.bmg


@pytest.fixture
def saved_model_path(tmp_path, test_batch, sample_targets):
    """Train a simple model and save it for conversion testing."""
    bmg, *_ = test_batch
    mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
    agg = MeanAggregation()
    pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
    model = MPNN(mp, agg, pred, batch_norm=False)
    model.eval()

    path = tmp_path / "test_model.pt"
    save_model(path, model)
    return path


def _build_pyg_input(bmg: BatchMolGraph):
    """Build PyG-compatible input tensors from a BatchMolGraph."""
    from torch_geometric.data import Batch

    x = bmg.V
    edge_index = bmg.edge_index
    edge_attr = bmg.E

    batch = Batch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        rev_edge_index=bmg.rev_edge_index,
        batch=bmg.batch,
    )
    return batch


class TestConvertToPyGBasic:
    """Test that conversion produces valid models."""

    def test_convert_returns_model(self, saved_model_path):
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        model = convert_model_to_pyg(saved_model_path)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_convert_file_creates_output(self, saved_model_path, tmp_path):
        from chemprop.utils.convert_to_pyg import convert_model_file_to_pyg

        output_path = tmp_path / "converted_pyg.pt"
        convert_model_file_to_pyg(saved_model_path, output_path)
        assert output_path.exists()

    def test_convert_file_saves_valid_checkpoint(self, saved_model_path, tmp_path):
        from chemprop.utils.convert_to_pyg import convert_model_file_to_pyg

        output_path = tmp_path / "converted_pyg.pt"
        convert_model_file_to_pyg(saved_model_path, output_path)

        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert "hyper_parameters" in checkpoint
        assert "state_dict" in checkpoint

    def test_load_pyg_model(self, saved_model_path, tmp_path):
        from chemprop.utils.convert_to_pyg import (
            convert_model_file_to_pyg,
            load_pyg_model,
        )

        output_path = tmp_path / "converted_pyg.pt"
        convert_model_file_to_pyg(saved_model_path, output_path)
        model = load_pyg_model(output_path)
        assert model is not None


class TestConvertToPyGEquivalence:
    """Test that PyG models produce same predictions as Chemprop models."""

    def test_bond_mp_regression_equivalence(self, test_batch, tmp_path):
        """Bond message passing + regression: PyG predictions match Chemprop."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "bond_mp.pt"
        save_model(path, model)

        # Get Chemprop predictions
        chemprop_pred = model(bmg)

        # Convert and get PyG predictions
        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_atom_mp_regression_equivalence(self, test_batch, tmp_path):
        """Atom message passing + regression: PyG predictions match Chemprop."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.AtomMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "atom_mp.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_sum_aggregation_equivalence(self, test_batch, tmp_path):
        """Sum aggregation produces matching predictions."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = SumAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "sum_agg.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_norm_aggregation_equivalence(self, test_batch, tmp_path):
        """Norm aggregation produces matching predictions."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = NormAggregation(norm=50.0)
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "norm_agg.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_batch_norm_equivalence(self, test_batch, tmp_path):
        """Batch norm enabled: PyG predictions match Chemprop (in eval mode)."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=True)
        model.eval()

        path = tmp_path / "batch_norm.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-4)

    def test_classification_equivalence(self, test_batch, tmp_path):
        """Binary classification: PyG predictions match Chemprop."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = nn.BinaryClassificationFFN(
            input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1
        )
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "classification.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_multitask_equivalence(self, test_batch, tmp_path):
        """Multi-task regression: PyG predictions match Chemprop."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        n_tasks = 3
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = nn.RegressionFFN(
            input_dim=mp.output_dim, n_tasks=n_tasks, hidden_dim=64, n_layers=1
        )
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "multitask.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert chemprop_pred.shape == pyg_pred.shape
        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_undirected_equivalence(self, test_batch, tmp_path):
        """Undirected message passing: PyG predictions match Chemprop."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(
            d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2, undirected=True
        )
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "undirected.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_different_activations(self, test_batch, tmp_path):
        """Test with tanh activation."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(
            d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2, activation="tanh"
        )
        agg = MeanAggregation()
        pred = nn.RegressionFFN(
            input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1, activation="tanh"
        )
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "tanh.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_deeper_model_equivalence(self, test_batch, tmp_path):
        """Deeper message passing (depth=5) and deeper FFN (n_layers=3)."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=5)
        agg = MeanAggregation()
        pred = nn.RegressionFFN(
            input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=3, dropout=0.1
        )
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "deeper.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)


class TestConvertToPyGCLI:
    """Test CLI conversion via the convert subcommand."""

    def test_cli_conversion_to_pyg(self, saved_model_path, tmp_path, capsys):
        """Test that 'chemprop convert --conversion to_pyg' works."""
        from chemprop.cli.convert import ConvertSubcommand

        output_path = tmp_path / "cli_converted_pyg.pt"
        args = Namespace(
            conversion="to_pyg",
            input_path=saved_model_path,
            output_path=output_path,
        )
        ConvertSubcommand.func(args)

        assert output_path.exists()

        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert "state_dict" in checkpoint
        assert "hyper_parameters" in checkpoint

    def test_cli_conversion_default_output_path(self, saved_model_path, tmp_path, monkeypatch):
        """Test default output path generation."""
        from chemprop.cli.convert import ConvertSubcommand

        monkeypatch.chdir(tmp_path)
        args = Namespace(
            conversion="to_pyg",
            input_path=saved_model_path,
            output_path=None,
        )
        ConvertSubcommand.func(args)

        expected = tmp_path / (saved_model_path.stem + "_pyg.pt")
        assert expected.exists()

    def test_cli_requires_pyg_dependency(self, saved_model_path, tmp_path, monkeypatch):
        """Test that the to_pyg conversion option is available."""
        from chemprop.cli.convert import ConvertSubcommand

        output_path = tmp_path / "cli_pyg_dep.pt"
        args = Namespace(
            conversion="to_pyg",
            input_path=saved_model_path,
            output_path=output_path,
        )
        # Should succeed since torch_geometric is installed
        ConvertSubcommand.func(args)
        assert output_path.exists()


class TestConvertToPyGModelProperties:
    """Test properties of converted PyG models."""

    def test_message_passing_type_bond(self, test_batch, tmp_path):
        from chemprop.utils.convert_to_pyg import (
            ChempropBondMessagePassingPyG,
            convert_model_to_pyg,
        )

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)

        path = tmp_path / "test_bond.pt"
        save_model(path, model)

        pyg_model = convert_model_to_pyg(path)
        assert isinstance(pyg_model.message_passing, ChempropBondMessagePassingPyG)

    def test_message_passing_type_atom(self, test_batch, tmp_path):
        from chemprop.utils.convert_to_pyg import (
            ChempropAtomMessagePassingPyG,
            convert_model_to_pyg,
        )

        bmg, *_ = test_batch
        mp = nn.AtomMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)

        path = tmp_path / "test_atom.pt"
        save_model(path, model)

        pyg_model = convert_model_to_pyg(path)
        assert isinstance(pyg_model.message_passing, ChempropAtomMessagePassingPyG)

    def test_aggregation_type_preserved(self, test_batch, tmp_path):
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)

        for agg, expected_type in [
            (MeanAggregation(), "mean"),
            (SumAggregation(), "sum"),
            (NormAggregation(norm=50.0), "norm"),
        ]:
            model = MPNN(mp, agg, pred, batch_norm=False)
            path = tmp_path / f"test_{expected_type}.pt"
            save_model(path, model)

            pyg_model = convert_model_to_pyg(path)
            assert pyg_model.agg_type == expected_type

    def test_model_eval_mode(self, saved_model_path):
        """Converted model should work in both train and eval mode."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        model = convert_model_to_pyg(saved_model_path)

        model.eval()
        assert not model.training

        model.train()
        assert model.training


class TestConvertToPyGErrors:
    """Test that unsupported model types raise proper errors."""

    def test_error_multicomponent(self, saved_model_path, tmp_path):
        """MulticomponentMPNN raises RuntimeError."""
        import torch
        from chemprop.nn.message_passing.multi import MulticomponentMessagePassing
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        checkpoint = torch.load(saved_model_path, map_location="cpu", weights_only=False)
        checkpoint["hyper_parameters"]["message_passing"]["cls"] = MulticomponentMessagePassing
        path = tmp_path / "multicomponent.pt"
        torch.save(checkpoint, path)

        with pytest.raises(RuntimeError, match="MulticomponentMPNN"):
            convert_model_to_pyg(path)

    def test_error_mol_atom_bond_cls(self, saved_model_path, tmp_path):
        """MolAtomBondMPNN detected via cls name raises RuntimeError."""
        import torch
        from chemprop.nn.message_passing.proto import MABMessagePassing
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        checkpoint = torch.load(saved_model_path, map_location="cpu", weights_only=False)
        checkpoint["hyper_parameters"]["message_passing"]["cls"] = MABMessagePassing
        path = tmp_path / "mab.pt"
        torch.save(checkpoint, path)

        with pytest.raises(RuntimeError, match="MolAtomBondMPNN"):
            convert_model_to_pyg(path)

    def test_error_mol_atom_bond_predictors(self, saved_model_path, tmp_path):
        """MolAtomBondMPNN detected via mol_predictor key raises RuntimeError."""
        import torch
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        checkpoint = torch.load(saved_model_path, map_location="cpu", weights_only=False)
        checkpoint["hyper_parameters"]["mol_predictor"] = {"cls": "RegressionFFN"}
        path = tmp_path / "mab_predictors.pt"
        torch.save(checkpoint, path)

        with pytest.raises(RuntimeError, match="MolAtomBondMPNN"):
            convert_model_to_pyg(path)


class TestConvertToPyGTransforms:
    """Test GraphTransform, V_d_transform, and X_d_transform support."""

    def test_graph_transform_equivalence(self, test_batch, tmp_path):
        """Model with GraphTransform produces matching predictions."""
        from chemprop.nn.transforms import GraphTransform, ScaleTransform
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        d_v, d_e = bmg.V.shape[1], bmg.E.shape[1]

        V_mean = torch.randn(d_v)
        V_scale = torch.abs(torch.randn(d_v)) + 0.1
        E_mean = torch.randn(d_e)
        E_scale = torch.abs(torch.randn(d_e)) + 0.1

        V_transform = ScaleTransform(V_mean.tolist(), V_scale.tolist())
        E_transform = ScaleTransform(E_mean.tolist(), E_scale.tolist())
        graph_transform = GraphTransform(V_transform, E_transform)

        mp = nn.BondMessagePassing(
            d_v=d_v, d_e=d_e, d_h=64, depth=2, graph_transform=graph_transform
        )
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "graph_transform.pt"
        save_model(path, model)

        # Save original V/E before Chemprop forward (GraphTransform modifies bmg in-place)
        V_orig = bmg.V.clone()
        E_orig = bmg.E.clone()

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                V_orig, bmg.edge_index, E_orig, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_v_d_transform_equivalence(self, test_batch, tmp_path):
        """Model with V_d_transform produces matching predictions."""
        from chemprop.nn.transforms import ScaleTransform
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        d_vd = 5
        V_d = torch.randn(bmg.V.shape[0], d_vd)

        vd_mean = torch.randn(d_vd)
        vd_scale = torch.abs(torch.randn(d_vd)) + 0.1
        V_d_transform = ScaleTransform(vd_mean.tolist(), vd_scale.tolist())

        mp = nn.BondMessagePassing(
            d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2,
            d_vd=d_vd, V_d_transform=V_d_transform,
        )
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "v_d_transform.pt"
        save_model(path, model)

        chemprop_pred = model(bmg, V_d=V_d)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch, v_d=V_d
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)

    def test_x_d_transform_equivalence(self, test_batch, tmp_path):
        """Model with X_d_transform produces matching predictions."""
        from chemprop.nn.transforms import ScaleTransform
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        num_mols = int(bmg.batch.max().item()) + 1
        d_xd = 10
        X_d = torch.randn(num_mols, d_xd)

        xd_mean = torch.randn(d_xd)
        xd_scale = torch.abs(torch.randn(d_xd)) + 0.1
        X_d_transform = ScaleTransform(xd_mean.tolist(), xd_scale.tolist())

        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = nn.RegressionFFN(input_dim=mp.output_dim + d_xd, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False, X_d_transform=X_d_transform)
        model.eval()

        path = tmp_path / "x_d_transform.pt"
        save_model(path, model)

        chemprop_pred = model(bmg, X_d=X_d)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch, X_d=X_d
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)


class TestConvertToPyGAttentiveAgg:
    """Test AttentiveAggregation support."""

    def test_attentive_aggregation_equivalence(self, test_batch, tmp_path):
        """Model with AttentiveAggregation produces matching predictions."""
        from chemprop.nn.agg import AttentiveAggregation
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = AttentiveAggregation(output_size=mp.output_dim)
        pred = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "attentive.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)


class TestConvertToPyGSpectralFFN:
    """Test SpectralFFN predictor support."""

    def test_spectral_predictor_equivalence(self, test_batch, tmp_path):
        """Model with SpectralFFN produces matching predictions."""
        from chemprop.nn.predictors import SpectralFFN
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2)
        agg = MeanAggregation()
        pred = SpectralFFN(input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1)
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "spectral.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert chemprop_pred.shape == pyg_pred.shape
        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)


class TestConvertToPyGPReLU:
    """Test PReLU activation support."""

    def test_prelu_activation_equivalence(self, test_batch, tmp_path):
        """Model with PReLU activation produces matching predictions."""
        from chemprop.utils.convert_to_pyg import convert_model_to_pyg

        bmg, *_ = test_batch
        mp = nn.BondMessagePassing(
            d_v=bmg.V.shape[1], d_e=bmg.E.shape[1], d_h=64, depth=2, activation="prelu"
        )
        agg = MeanAggregation()
        pred = nn.RegressionFFN(
            input_dim=mp.output_dim, n_tasks=1, hidden_dim=64, n_layers=1, activation="prelu"
        )
        model = MPNN(mp, agg, pred, batch_norm=False)
        model.eval()

        path = tmp_path / "prelu.pt"
        save_model(path, model)

        chemprop_pred = model(bmg)

        pyg_model = convert_model_to_pyg(path)
        pyg_model.eval()
        with torch.no_grad():
            pyg_pred = pyg_model(
                bmg.V, bmg.edge_index, bmg.E, bmg.rev_edge_index, bmg.batch
            )

        assert torch.allclose(chemprop_pred, pyg_pred, atol=1e-5)
