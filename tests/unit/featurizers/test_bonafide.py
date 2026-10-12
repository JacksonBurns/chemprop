"""Unit tests for the BONAFIDE (atom-bond-featurizer) chemprop integration.

BONAFIDE is an optional, heavy dependency. These tests are skipped when it is not installed
(see ``pytest.importorskip``), mirroring how chemprop handles optional featurizer backends.
"""

import pickle
import warnings

import numpy as np
import pytest
from rdkit import Chem

pytest.importorskip("bonafide")

from chemprop.data.molgraph import MolGraph  # noqa: E402
from chemprop.featurizers import (  # noqa: E402
    BonafideAtomFeaturizer,
    BonafideBondFeaturizer,
    BonafideMolGraphFeaturizer,
    MultiHotAtomFeaturizer,
    MultiHotBondFeaturizer,
    bonafide_feature_indices,
)
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer  # noqa: E402


@pytest.fixture
def bonafide_atom_indices():
    return bonafide_feature_indices(
        "atom", origin="rdkit", dimensionality="2D", requires_electronic_structure_data=False
    )


@pytest.fixture
def bonafide_bond_indices():
    return bonafide_feature_indices(
        "bond", origin="rdkit", dimensionality="2D", requires_electronic_structure_data=False
    )


@pytest.fixture
def bonafide_featurizer(bonafide_atom_indices, bonafide_bond_indices):
    """The default featurizer: BONAFIDE features only (no base multi-hot graph)."""
    return BonafideMolGraphFeaturizer(
        bonafide_atom_indices=bonafide_atom_indices,
        bonafide_bond_indices=bonafide_bond_indices,
    )


@pytest.fixture
def combined_featurizer(bonafide_atom_indices, bonafide_bond_indices):
    """Base multi-hot graph + BONAFIDE features concatenated."""
    return BonafideMolGraphFeaturizer(
        atom_featurizer=MultiHotAtomFeaturizer.v2(),
        bond_featurizer=MultiHotBondFeaturizer(),
        bonafide_atom_indices=bonafide_atom_indices,
        bonafide_bond_indices=bonafide_bond_indices,
    )


def test_feature_indices_are_numeric(bonafide_atom_indices, bonafide_bond_indices):
    assert all(isinstance(i, int) for i in bonafide_atom_indices)
    assert all(isinstance(i, int) for i in bonafide_bond_indices)
    assert len(bonafide_atom_indices) > 0
    assert len(bonafide_bond_indices) > 0


def test_default_is_bonafide_only(mol, bonafide_featurizer, bonafide_atom_indices, bonafide_bond_indices):
    # The default uses BONAFIDE features exclusively (no base multi-hot graph).
    assert bonafide_featurizer.atom_featurizer is None
    assert bonafide_featurizer.bond_featurizer is None
    assert bonafide_featurizer.shape == (
        len(bonafide_atom_indices),
        len(bonafide_bond_indices),
    )


def test_shape_extends_base(mol, combined_featurizer, bonafide_atom_indices, bonafide_bond_indices):
    base_atom, base_bond = SimpleMoleculeMolGraphFeaturizer().shape
    assert combined_featurizer.shape == (
        base_atom + len(bonafide_atom_indices),
        base_bond + len(bonafide_bond_indices),
    )
    # shape property agrees with the atom_fdim / bond_fdim attributes
    assert combined_featurizer.shape == (combined_featurizer.atom_fdim, combined_featurizer.bond_fdim)


def test_V_E_shape(mol, bonafide_featurizer: BonafideMolGraphFeaturizer):
    mg = bonafide_featurizer(mol)
    assert isinstance(mg, MolGraph)
    n_a = mol.GetNumAtoms()
    n_b = mol.GetNumBonds()
    assert mg.V.shape == (n_a, bonafide_featurizer.atom_fdim)
    assert mg.E.shape == (2 * n_b, bonafide_featurizer.bond_fdim)
    assert mg.edge_index.shape == (2, 2 * n_b)
    assert mg.rev_edge_index.shape == (2 * n_b,)


def test_base_graph_is_prefix(mol, combined_featurizer, bonafide_atom_indices, bonafide_bond_indices):
    base_atom, base_bond = SimpleMoleculeMolGraphFeaturizer().shape
    base_mg = SimpleMoleculeMolGraphFeaturizer()(mol)
    bf_mg = combined_featurizer(mol)

    np.testing.assert_allclose(bf_mg.V[:, :base_atom], base_mg.V, atol=1e-6)
    np.testing.assert_allclose(bf_mg.E[:, :base_bond], base_mg.E, atol=1e-6)
    np.testing.assert_array_equal(bf_mg.edge_index, base_mg.edge_index)
    np.testing.assert_array_equal(bf_mg.rev_edge_index, base_mg.rev_edge_index)


def test_bonafide_block_nontrivial(mol, combined_featurizer, bonafide_atom_indices, bonafide_bond_indices):
    base_atom, base_bond = SimpleMoleculeMolGraphFeaturizer().shape
    mg = combined_featurizer(mol)
    assert np.isfinite(mg.V).all()
    assert np.isfinite(mg.E).all()
    assert np.count_nonzero(mg.V[:, base_atom:]) > 0
    assert np.count_nonzero(mg.E[:, base_bond:]) > 0


def test_bonafide_only_graph(mol, bonafide_atom_indices, bonafide_bond_indices):
    bf = BonafideMolGraphFeaturizer(
        atom_featurizer=None,
        bond_featurizer=None,
        bonafide_atom_indices=bonafide_atom_indices,
        bonafide_bond_indices=bonafide_bond_indices,
    )
    assert bf.shape == (len(bonafide_atom_indices), len(bonafide_bond_indices))
    mg = bf(mol)
    assert mg.V.shape == (mol.GetNumAtoms(), len(bonafide_atom_indices))
    assert mg.E.shape == (2 * mol.GetNumBonds(), len(bonafide_bond_indices))


def test_atom_bond_vector_featurizers(mol, bonafide_atom_indices, bonafide_bond_indices):
    af = BonafideAtomFeaturizer(feature_indices=bonafide_atom_indices)
    bf = BonafideBondFeaturizer(feature_indices=bonafide_bond_indices)

    assert len(af) == len(bonafide_atom_indices)
    assert len(bf) == len(bonafide_bond_indices)

    atom = mol.GetAtomWithIdx(0)
    bond = mol.GetBondWithIdx(0)
    v_a = af(atom)
    v_b = bf(bond)
    assert v_a.shape == (len(bonafide_atom_indices),)
    assert v_b.shape == (len(bonafide_bond_indices),)

    # Per-atom/per-bond rows must match the corresponding rows of the whole-molecule matrix.
    g = BonafideMolGraphFeaturizer(
        atom_featurizer=None,
        bond_featurizer=None,
        bonafide_atom_indices=bonafide_atom_indices,
        bonafide_bond_indices=bonafide_bond_indices,
    )
    mg = g(mol)
    np.testing.assert_allclose(v_a, mg.V[0], atol=1e-6)
    np.testing.assert_allclose(v_b, mg.E[0], atol=1e-6)

    # None handling returns a zero vector
    np.testing.assert_array_equal(af(None), np.zeros(len(bonafide_atom_indices)))
    np.testing.assert_array_equal(bf(None), np.zeros(len(bonafide_bond_indices)))


def test_extra_features_contract(mol, bonafide_featurizer, bonafide_atom_indices, bonafide_bond_indices):
    n_a, n_b = mol.GetNumAtoms(), mol.GetNumBonds()
    a_extra = np.random.rand(n_a, 3)
    b_extra = np.random.rand(n_b, 2)
    mg = bonafide_featurizer(mol, atom_features_extra=a_extra, bond_features_extra=b_extra)
    assert mg.V.shape == (n_a, bonafide_featurizer.atom_fdim + 3)
    assert mg.E.shape == (2 * n_b, bonafide_featurizer.bond_fdim + 2)
    np.testing.assert_array_equal(mg.V[:, -3:], a_extra)
    # bond extras are duplicated per directed edge
    np.testing.assert_array_equal(mg.E[0::2, -2:], b_extra)
    np.testing.assert_array_equal(mg.E[1::2, -2:], b_extra)

    with pytest.raises(ValueError):
        bonafide_featurizer(mol, atom_features_extra=np.random.rand(n_a + 1, 3))
    with pytest.raises(ValueError):
        bonafide_featurizer(mol, bond_features_extra=np.random.rand(n_b + 1, 2))


def test_picklable(mol, bonafide_featurizer):
    # MolGraphCache uses a multiprocess.Pool (fork) when n_workers>=2, so the featurizer
    # must pickle cleanly and reproduce identical features after unpickling.
    restored = pickle.loads(pickle.dumps(bonafide_featurizer))
    mg1, mg2 = bonafide_featurizer(mol), restored(mol)
    np.testing.assert_array_equal(mg1.V, mg2.V)
    np.testing.assert_array_equal(mg1.E, mg2.E)
    np.testing.assert_array_equal(mg1.edge_index, mg2.edge_index)
    np.testing.assert_array_equal(mg1.rev_edge_index, mg2.rev_edge_index)


def test_from_filters(mol):
    bf = BonafideMolGraphFeaturizer.from_filters(
        atom_filters={"origin": "rdkit", "dimensionality": "2D", "requires_electronic_structure_data": False},
        bond_filters={"origin": "rdkit", "dimensionality": "2D", "requires_electronic_structure_data": False},
    )
    mg = bf(mol)
    assert mg.V.shape[1] == bf.atom_fdim
    assert mg.E.shape[1] == bf.bond_fdim


def test_zero_bond_fallback(bonafide_atom_indices, bonafide_bond_indices):
    # A zero-bond atom (e.g. "C") has no bond data, which some BONAFIDE features require.
    # The featurizer must degrade to a zero vector rather than crash the pipeline.
    bf = BonafideMolGraphFeaturizer(
        bonafide_atom_indices=bonafide_atom_indices,
        bonafide_bond_indices=bonafide_bond_indices,
    )
    mol = Chem.MolFromSmiles("C")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mg = bf(mol)
    assert mg.V.shape == (1, len(bonafide_atom_indices))
    assert mg.E.shape == (0, len(bonafide_bond_indices))
    assert np.isfinite(mg.V).all()
