"""BONAFIDE (``atom-bond-featurizer``) integration for chemprop.

This module exposes the `BONAFIDE <https://github.com/MolecularAI/atom-bond-featurizer>`__
atom/bond featurizer as a chemprop :class:`~chemprop.featurizers.base.GraphFeaturizer`,
modelled on the existing :class:`~chemprop.featurizers.molgraph.molecule.SimpleMoleculeMolGraphFeaturizer`.

BONAFIDE has a *stateful, whole-molecule* API (``read_input`` -> ``featurize_atoms`` ->
``return_atom_features`` returning a per-atom DataFrame), which is the opposite of chemprop's
*stateless, per-atom* :class:`~chemprop.featurizers.base.VectorFeaturizer` contract. The
:class:`BonafideMolGraphFeaturizer` bridges the two: for each molecule it computes the BONAFIDE
atom and bond feature matrices once and uses them as the node/edge features of the graph. By
default it uses **only** BONAFIDE features (no base multi-hot chemprop graph); an optional base
``atom_featurizer``/``bond_featurizer`` can be supplied to concatenate
``[base_atom_feats | bonafide_atom_feats]`` per node and
``[base_bond_feats | bonafide_bond_feats]`` per edge.

The BONAFIDE :class:`~bonafide.AtomBondFeaturizer` instance is created lazily, once per OS
process, and excluded from pickling. This matters because chemprop's
:class:`~chemprop.featurizers.molgraph.cache.MolGraphCache` featurizes in parallel via a
``multiprocess.Pool`` (fork): a shared parent-process instance would be raced on by every
worker, so each worker transparently materialises its own instance on first use.
"""

from __future__ import annotations

import json
import os
import uuid
import warnings
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import GraphFeaturizer, VectorFeaturizer
from rdkit.Chem.rdchem import Atom, Bond

__all__ = [
    "bonafide_feature_indices",
    "BonafideEngine",
    "BonafideAtomFeaturizer",
    "BonafideBondFeaturizer",
    "BonafideMolGraphFeaturizer",
]

# Data types BONAFIDE reports that are usable directly as numeric graph features.
_NUMERIC_TYPES = ("float", "int", "bool")


def _bonafide_feature_catalog() -> dict[str, dict[str, Any]]:
    """Load BONAFIDE's feature catalog (``_feature_info.json``) without importing ``bonafide``.

    Reading the JSON directly (rather than instantiating an
    :class:`~bonafide.AtomBondFeaturizer` just to call ``list_*_features``) keeps feature
    selection side-effect free and avoids the BONAFIDE log-file creation at import time.
    """
    import bonafide

    path = os.path.join(os.path.dirname(bonafide.__file__), "_feature_info.json")
    with open(path) as fh:
        return json.load(fh)


def bonafide_feature_indices(feature_type: str, **filters: Any) -> list[int]:
    """Resolve a set of BONAFIDE feature filters into a sorted list of integer feature indices.

    Parameters
    ----------
    feature_type : str
        Either ``"atom"`` or ``"bond"``.
    **filters : Any
        Any of the columns of the BONAFIDE feature catalog, e.g. ``origin``,
        ``dimensionality``, ``data_type``, ``requires_electronic_structure_data``,
        ``requires_bond_data``, ``requires_charge``, ``requires_multiplicity``. ``data_type``
        defaults to the numeric types (``float``/``int``/``bool``) since string features cannot
        be used directly as graph features.

    Returns
    -------
    list[int]
        The sorted BONAFIDE integer indices matching all provided filters.
    """
    if feature_type not in ("atom", "bond"):
        raise ValueError(f"feature_type must be 'atom' or 'bond', got {feature_type!r}")

    filters.setdefault("data_type", _NUMERIC_TYPES)
    catalog = _bonafide_feature_catalog()

    indices: list[int] = []
    for key, meta in catalog.items():
        if meta["feature_type"] != feature_type:
            continue
        for name, wanted in filters.items():
            got = meta[name]
            if isinstance(wanted, (list, tuple, set)):
                match = got in wanted
            else:
                match = got == wanted
            if not match:
                break
        else:
            indices.append(int(key))

    return sorted(indices)


class BonafideEngine:
    """A lazily-initialised, per-process, picklable wrapper around
    :class:`~bonafide.AtomBondFeaturizer`.

    The BONAFIDE instance is created on first use in the current process (keyed by PID) and is
    dropped from the pickled state, so each ``multiprocess`` worker materialises its own
    independent instance after forking.

    Parameters
    ----------
    atom_feature_indices : Sequence[int]
        The BONAFIDE integer atom-feature indices to compute.
    bond_feature_indices : Sequence[int]
        The BONAFIDE integer bond-feature indices to compute.
    log_file_name : str | None
        Base name for the BONAFIDE log file. A per-process suffix is appended. If ``None`` a
        default name under the current working directory is used.
    """

    def __init__(
        self,
        atom_feature_indices: Sequence[int],
        bond_feature_indices: Sequence[int],
        log_file_name: str | None = None,
    ):
        self._atom_feature_indices = list(atom_feature_indices)
        self._bond_feature_indices = list(bond_feature_indices)
        self._log_file_name = log_file_name
        self._bonafide = None
        self._bonafide_pid = None

    # -- pickling -----------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_bonafide"] = None
        state["_bonafide_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # -- lazy instance ------------------------------------------------------
    def _bonafide_instance(self):
        from bonafide import AtomBondFeaturizer  # imported lazily so chemprop loads without it

        pid = os.getpid()
        if self._bonafide is None or self._bonafide_pid != pid:
            base = self._log_file_name or "bonafide_chemprop.log"
            log_name = f"{base}.{pid}"
            # BONAFIDE refuses to write to an existing log file; start fresh each process.
            if os.path.exists(log_name):
                os.remove(log_name)
            self._bonafide = AtomBondFeaturizer(log_file_name=log_name)
            self._bonafide_pid = pid
        return self._bonafide

    # -- feature computation ------------------------------------------------
    def featurize(self, mol: Mol) -> tuple[np.ndarray, np.ndarray]:
        """Compute the BONAFIDE atom and bond feature matrices for ``mol``.

        Returns
        -------
        (atom_feats, bond_feats) : tuple[np.ndarray, np.ndarray]
            Arrays of shape ``(n_atoms, len(atom_feature_indices))`` and
            ``(n_bonds, len(bond_feature_indices))`` with rows aligned to RDKit atom/bond
            indices.
        """
        bon = self._bonafide_instance()
        namespace = f"chemprop_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        # BONAFIDE stores its results as RDKit atom/bond properties on the molecule it is given.
        # Featurizing a copy keeps the caller's molecule pristine and avoids stale properties
        # leaking between featurize() calls on the same object.
        mol = Chem.Mol(mol)

        n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()

        try:
            bon.read_input(input_value=mol, namespace=namespace, input_format="mol_object")

            if self._atom_feature_indices:
                bon.featurize_atoms(atom_indices="all", feature_indices=self._atom_feature_indices)
                at = bon.return_atom_features()
                atom_feats = at.reindex(range(n_atoms)).fillna(0.0).to_numpy(dtype=np.float32)
            else:
                atom_feats = np.zeros((n_atoms, 0), dtype=np.float32)

            if self._bond_feature_indices:
                bon.featurize_bonds(bond_indices="all", feature_indices=self._bond_feature_indices)
                bo = bon.return_bond_features()
                bond_feats = bo.reindex(range(n_bonds)).fillna(0.0).to_numpy(dtype=np.float32)
            else:
                bond_feats = np.zeros((n_bonds, 0), dtype=np.float32)

            return atom_feats, bond_feats

        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never break the pipeline
            # Some molecules are outside a given BONAFIDE feature set's domain (e.g. a zero-bond
            # atom like "C" has no bond data for bond-dependent features). Fall back to a zero
            # vector so one odd molecule cannot take down a whole dataset; warn once.
            if not getattr(self, "_warned", False):
                warnings.warn(
                    f"BONAFIDE could not featurize {n_atoms}-atom molecule ({type(exc).__name__}: "
                    f"{str(exc)[:80]}...); using zero features for it. Other molecules are "
                    "unaffected. This warning is shown only once.",
                    stacklevel=2,
                )
                self._warned = True
            atom_feats = np.zeros((n_atoms, len(self._atom_feature_indices)), dtype=np.float32)
            bond_feats = np.zeros((n_bonds, len(self._bond_feature_indices)), dtype=np.float32)
            return atom_feats, bond_feats


def _default_atom_indices() -> list[int]:
    """Default: RDKit 2D numeric atom features (fast, no external programs required)."""
    return bonafide_feature_indices(
        "atom", origin="rdkit", dimensionality="2D", requires_electronic_structure_data=False
    )


def _default_bond_indices() -> list[int]:
    """Default: RDKit 2D numeric bond features (fast, no external programs required)."""
    return bonafide_feature_indices(
        "bond", origin="rdkit", dimensionality="2D", requires_electronic_structure_data=False
    )


class BonafideAtomFeaturizer(VectorFeaturizer[Atom]):
    """A :class:`VectorFeaturizer` that returns the BONAFIDE feature vector of a single atom.

    BONAFIDE computes features for the whole molecule, so this featurizer recovers the atom's
    parent molecule, computes the BONAFIDE atom matrix once, and returns the requested atom's
    row. For featurising an entire molecule, prefer
    :class:`BonafideMolGraphFeaturizer`, which computes the matrix a single time.

    Parameters
    ----------
    feature_indices : Sequence[int] | None
        The BONAFIDE integer atom-feature indices. Defaults to the RDKit 2D numeric set.
    engine : BonafideEngine | None
        A shared engine. If ``None`` a per-instance engine is created.
    log_file_name : str | None
        Base name for the BONAFIDE log file (see :class:`BonafideEngine`).
    """

    def __init__(
        self,
        feature_indices: Sequence[int] | None = None,
        engine: BonafideEngine | None = None,
        log_file_name: str | None = None,
    ):
        self._feature_indices = list(feature_indices) if feature_indices is not None else _default_atom_indices()
        self._engine = engine if engine is not None else BonafideEngine(
            self._feature_indices, [], log_file_name=log_file_name
        )

    def __len__(self) -> int:
        return len(self._feature_indices)

    def __call__(self, a: Atom | None) -> np.ndarray:
        if a is None:
            return np.zeros(len(self), dtype=np.float32)
        mol = a.GetOwningMol()
        atom_feats, _ = self._engine.featurize(mol)
        return atom_feats[a.GetIdx()]


class BonafideBondFeaturizer(VectorFeaturizer[Bond]):
    """A :class:`VectorFeaturizer` that returns the BONAFIDE feature vector of a single bond.

    See :class:`BonafideAtomFeaturizer` for the whole-molecule caveat.

    Parameters
    ----------
    feature_indices : Sequence[int] | None
        The BONAFIDE integer bond-feature indices. Defaults to the RDKit 2D numeric set.
    engine : BonafideEngine | None
        A shared engine. If ``None`` a per-instance engine is created.
    log_file_name : str | None
        Base name for the BONAFIDE log file (see :class:`BonafideEngine`).
    """

    def __init__(
        self,
        feature_indices: Sequence[int] | None = None,
        engine: BonafideEngine | None = None,
        log_file_name: str | None = None,
    ):
        self._feature_indices = list(feature_indices) if feature_indices is not None else _default_bond_indices()
        self._engine = engine if engine is not None else BonafideEngine(
            [], self._feature_indices, log_file_name=log_file_name
        )

    def __len__(self) -> int:
        return len(self._feature_indices)

    def __call__(self, b: Bond | None) -> np.ndarray:
        if b is None:
            return np.zeros(len(self), dtype=np.float32)
        mol = b.GetBeginAtom().GetOwningMol()
        _, bond_feats = self._engine.featurize(mol)
        return bond_feats[b.GetIdx()]


class BonafideMolGraphFeaturizer(GraphFeaturizer[Mol]):
    r"""A chemprop :class:`GraphFeaturizer` that represents molecules purely with BONAFIDE
    atom/bond features.

    For each molecule the BONAFIDE atom and bond feature matrices are computed once and used
    directly as the node/edge features of the resulting
    :class:`~chemprop.data.molgraph.MolGraph`, so the graph carries

    * node features ``bonafide_atom_feats`` of width ``atom_fdim``, and
    * edge features ``bonafide_bond_feats`` of width ``bond_fdim``.

    By default **no** base multi-hot chemprop features are used - the model is featurized
    exclusively from BONAFIDE, which is useful for isolating how well the BONAFIDE features
    perform on their own. To layer a base multi-hot graph underneath (concatenated as
    ``[base | bonafide]``), pass an ``atom_featurizer``/``bond_featurizer``.

    Parameters
    ----------
    atom_featurizer : VectorFeaturizer[Atom] | None
        An optional base per-atom featurizer concatenated in front of the BONAFIDE atom
        features. Defaults to ``None`` (BONAFIDE-only node representation).
    bond_featurizer : VectorFeaturizer[Bond] | None
        An optional base per-bond featurizer concatenated in front of the BONAFIDE bond
        features. Defaults to ``None`` (BONAFIDE-only edge representation).
    bonafide_atom_indices : Sequence[int] | None
        BONAFIDE integer atom-feature indices (default: RDKit 2D numeric set).
    bonafide_bond_indices : Sequence[int] | None
        BONAFIDE integer bond-feature indices (default: RDKit 2D numeric set).
    log_file_name : str | None
        Base name for the BONAFIDE log file (see :class:`BonafideEngine`).
    """

    def __init__(
        self,
        atom_featurizer: VectorFeaturizer[Atom] | None = None,
        bond_featurizer: VectorFeaturizer[Bond] | None = None,
        bonafide_atom_indices: Sequence[int] | None = None,
        bonafide_bond_indices: Sequence[int] | None = None,
        log_file_name: str | None = None,
    ):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self._bonafide_atom_indices = (
            list(bonafide_atom_indices) if bonafide_atom_indices is not None else _default_atom_indices()
        )
        self._bonafide_bond_indices = (
            list(bonafide_bond_indices) if bonafide_bond_indices is not None else _default_bond_indices()
        )
        self._engine = BonafideEngine(
            self._bonafide_atom_indices,
            self._bonafide_bond_indices,
            log_file_name=log_file_name,
        )

        self._base_atom_fdim = len(self.atom_featurizer) if self.atom_featurizer is not None else 0
        self._base_bond_fdim = len(self.bond_featurizer) if self.bond_featurizer is not None else 0
        self.atom_fdim = self._base_atom_fdim + len(self._bonafide_atom_indices)
        self.bond_fdim = self._base_bond_fdim + len(self._bonafide_bond_indices)

    @property
    def shape(self) -> tuple[int, int]:
        """the feature dimension of the atoms and bonds, respectively, of the ``MolGraph``s
        generated by this featurizer"""
        return self.atom_fdim, self.bond_fdim

    @classmethod
    def from_filters(
        cls,
        atom_featurizer: VectorFeaturizer[Atom] | None = None,
        bond_featurizer: VectorFeaturizer[Bond] | None = None,
        atom_filters: dict[str, Any] | None = None,
        bond_filters: dict[str, Any] | None = None,
        log_file_name: str | None = None,
    ) -> "BonafideMolGraphFeaturizer":
        """Build a :class:`BonafideMolGraphFeaturizer` selecting BONAFIDE features by catalog
        filters (see :func:`bonafide_feature_indices`) instead of explicit integer indices."""
        atom_filters = atom_filters or {
            "origin": "rdkit",
            "dimensionality": "2D",
            "requires_electronic_structure_data": False,
        }
        bond_filters = bond_filters or {
            "origin": "rdkit",
            "dimensionality": "2D",
            "requires_electronic_structure_data": False,
        }
        return cls(
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            bonafide_atom_indices=bonafide_feature_indices("atom", **atom_filters),
            bonafide_bond_indices=bonafide_feature_indices("bond", **bond_filters),
            log_file_name=log_file_name,
        )

    def __call__(
        self,
        mol: Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        # Base node/edge features (or zeros for a BONAFIDE-only representation).
        if self.atom_featurizer is not None:
            V_base = (
                np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
                if n_atoms
                else np.zeros((1, self._base_atom_fdim), dtype=np.single)
            )
        else:
            V_base = np.zeros((n_atoms, 0), dtype=np.single) if n_atoms else np.zeros((1, 0), dtype=np.single)
        if self.bond_featurizer is not None:
            E_base = np.array([self.bond_featurizer(b) for b in mol.GetBonds()], dtype=np.single)
        else:
            E_base = np.zeros((n_bonds, 0), dtype=np.single)

        # BONAFIDE node/edge features (computed once for the whole molecule).
        A, B = self._engine.featurize(mol)

        # Concatenate base | bonafide along the feature axis. Guard on the number of feature
        # *columns* (not total size), so a molecule with zero atoms/bonds keeps the correct width.
        V = np.hstack((V_base, A)) if A.shape[1] else V_base
        E = np.hstack((E_base, B)) if B.shape[1] else E_base

        # Optional per-molecule extra features (same contract as SimpleMoleculeMolGraphFeaturizer).
        if atom_features_extra is not None:
            if len(atom_features_extra) != n_atoms:
                raise ValueError(
                    "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                    f" got: {n_atoms} and {len(atom_features_extra)}, respectively"
                )
            V = np.hstack((V, atom_features_extra))
        if bond_features_extra is not None:
            if len(bond_features_extra) != n_bonds:
                raise ValueError(
                    "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                    f" got: {n_bonds} and {len(bond_features_extra)}, respectively"
                )
            E = np.hstack((E, bond_features_extra))

        # Build directed edge index (each undirected bond -> two directed edges), matching
        # SimpleMoleculeMolGraphFeaturizer.
        edge_index = [[], []]
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

        # E is one row per undirected bond; expand to one row per directed edge.
        E_dir = np.repeat(E, repeats=2, axis=0)
        rev_edge_index = np.arange(E_dir.shape[0]).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, dtype=int)

        return MolGraph(V, E_dir, edge_index, rev_edge_index)
