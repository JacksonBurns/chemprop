import numpy as np
import pytest
import torch
from torch import Tensor

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolAtomBondGraph, MolGraph
from chemprop.data.collate import MolAtomBondTrainingBatch
from chemprop.models import MolAtomBondMPNN
from chemprop.nn import MSE, MABBondMessagePassing, MeanAggregation, RegressionFFN


@pytest.fixture
def mp():
    return MABBondMessagePassing()


@pytest.fixture
def agg():
    return MeanAggregation()


@pytest.fixture
def ffn():
    return RegressionFFN()


@pytest.fixture
def full_model(mp, agg, ffn):
    return MolAtomBondMPNN(
        message_passing=mp, agg=agg, mol_predictor=ffn, atom_predictor=ffn, bond_predictor=ffn
    )


@pytest.fixture
def partial_model(mp, agg, ffn):
    return MolAtomBondMPNN(
        message_passing=mp, agg=agg, mol_predictor=ffn, atom_predictor=None, bond_predictor=ffn
    )


def test_output_dimss(full_model, partial_model):
    assert full_model.output_dimss == (1, 1, 1)
    assert partial_model.output_dimss == (1, None, 1)


def test_n_taskss(full_model, partial_model):
    assert full_model.n_taskss == (1, 1, 1)
    assert partial_model.n_taskss == (1, None, 1)


def test_n_targetss(full_model, partial_model):
    assert full_model.n_targetss == (1, 1, 1)
    assert partial_model.n_targetss == (1, None, 1)


def test_criterions_lists(full_model, partial_model):
    assert all(isinstance(c, MSE) for c in full_model.criterions)
    assert isinstance(partial_model.criterions[0], MSE)
    assert partial_model.criterions[1] is None
    assert isinstance(partial_model.criterions[2], MSE)


def _make_dummy_batch(
    n_mols: int = 2, n_atoms: int = 3, n_bonds: int = 2
) -> tuple[BatchMolAtomBondGraph, Tensor, Tensor]:
    """Create a minimal BatchMolAtomBondGraph and descriptor tensors for testing."""
    mgs = []
    # Each bond has 2 directed edges, so E and edge_index have 2*n_bonds entries
    n_edges = 2 * n_bonds
    for _ in range(n_mols):
        V = np.random.randn(n_atoms, DEFAULT_ATOM_FDIM).astype(np.float32)
        E = np.random.randn(n_edges, DEFAULT_BOND_FDIM).astype(np.float32)
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        rev_edge_index = np.array([1, 0, 3, 2], dtype=np.int64)
        mgs.append(MolGraph(V, E, edge_index, rev_edge_index))

    bmg = BatchMolAtomBondGraph(mgs)
    V_d = torch.randn(n_mols * n_atoms, DEFAULT_HIDDEN_DIM)
    E_d = torch.randn(n_mols * n_edges, DEFAULT_HIDDEN_DIM)
    return bmg, V_d, E_d


def test_forward_pass_default_predictors(mp, agg):
    """Verify forward pass works with default RegressionFFN (input_dim=DEFAULT_HIDDEN_DIM).

    This tests that the bond predictor receives the correct input dimension (not doubled
    by edge concatenation), fixing the dimension mismatch reported in the issue.
    """
    ffn = RegressionFFN()
    model = MolAtomBondMPNN(
        message_passing=mp,
        agg=agg,
        mol_predictor=ffn,
        atom_predictor=ffn,
        bond_predictor=ffn,
    )

    bmg, _, _ = _make_dummy_batch()
    mol_pred, atom_pred, bond_pred = model(bmg)

    # Verify shapes are valid (no RuntimeError from dimension mismatch)
    assert mol_pred.ndim == 2
    assert atom_pred.ndim == 2
    assert bond_pred.ndim == 2
    # n_tasks=1 for default RegressionFFN
    assert mol_pred.shape[1] == 1
    assert atom_pred.shape[1] == 1
    assert bond_pred.shape[1] == 1


def test_fingerprint_dims_match_predictor_input(mp, agg):
    """Ensure bond fingerprint dimension equals the predictor's expected input_dim."""
    ffn = RegressionFFN()
    model = MolAtomBondMPNN(
        message_passing=mp,
        agg=agg,
        mol_predictor=ffn,
        atom_predictor=ffn,
        bond_predictor=ffn,
    )

    bmg, _, _ = _make_dummy_batch()
    H_g, H_v, H_e = model.fingerprint(bmg)

    assert H_g.shape[1] == model.mol_predictor.input_dim
    assert H_v.shape[1] == model.atom_predictor.input_dim
    assert H_e.shape[1] == model.bond_predictor.input_dim
