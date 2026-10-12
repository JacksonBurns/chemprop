#!/usr/bin/env python
"""
Demo: using the BONAFIDE (atom-bond-featurizer) featurizer with chemprop.

This script shows, end to end, how to:
  1. build a chemprop graph featurizer from BONAFIDE atom/bond features (BONAFIDE-only,
     i.e. *no* base multi-hot chemprop features - we want to see how well BONAFIDE alone does),
  2. fit a chemprop MPNN on a real benchmark (MoleculeNet ESOL lipophilicity, 1,144 molecules),
  3. report RMSE and Pearson R on train/val/test, with a train-mean baseline for reference.

Run it (in the `chemprop-dev` conda env, which has both chemprop and bonafide installed):

    python bonafide_esol_demo.py [path/to/esol.csv]

The default CSV is the ESOL set that ships with the `interpretable_aqueous_solubility` project.
The CSV just needs two columns: ``SMILES`` and ``logS``.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from rdkit import Chem

# --- chemprop building blocks -----------------------------------------------
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import (
    BonafideMolGraphFeaturizer,
    bonafide_feature_indices,
)
from chemprop.models import MPNN
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.predictors import RegressionFFN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# 1. Data                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    smiles: list[str]
    targets: np.ndarray  # shape (n, 1)


def load_esol(csv_path: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    # Be lenient about the exact target column name.
    target_col = "logS" if "logS" in df.columns else df.columns[1]
    df = df.dropna(subset=["SMILES", target_col])
    return df["SMILES"].tolist(), df[target_col].to_numpy(dtype=float).reshape(-1, 1)


def random_split(
    smiles: list[str], targets: np.ndarray, seed: int = 42
) -> tuple[Split, Split, Split]:
    n = len(smiles)
    idx = np.random.RandomState(seed).permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    tr, va, te = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
    return (
        Split([smiles[i] for i in tr], targets[tr]),
        Split([smiles[i] for i in va], targets[va]),
        Split([smiles[i] for i in te], targets[te]),
    )


def to_dataset(split: Split, featurizer: BonafideMolGraphFeaturizer, n_workers: int = 4) -> MoleculeDataset:
    data = [MoleculeDatapoint.from_smi(s, y=[t]) for s, t in zip(split.smiles, split.targets.ravel())]
    dset = MoleculeDataset(data=data, featurizer=featurizer, n_workers=n_workers)
    dset.cache = True  # featurize once, in parallel
    return dset


# --------------------------------------------------------------------------- #
# 2. Feature standardization (train-only stats, applied to all splits)         #
# --------------------------------------------------------------------------- #
def standardize_features(
    train_dset: MoleculeDataset, all_dsets: list[MoleculeDataset]
) -> None:
    """Z-score the node (V) and edge (E) features using statistics from the TRAIN set only.

    BONAFIDE features span very different scales (e.g. atomic mass ~12, logP contributions,
    bond orders), so standardizing gives the MPNN well-conditioned inputs. This mirrors the
    target scaling chemprop does and is a normal preprocessing step for a custom featurizer.

    ``MolGraphCache`` exposes only ``__getitem__`` (over its internal ``_mgs`` list), so we
    rebuild that list in place with standardized :class:`MolGraph`` copies.
    """
    # Stack all train node/edge rows to get per-feature mean/std.
    train_V = np.concatenate([g.V for g in train_dset.mg_cache._mgs])
    train_E = np.concatenate([g.E for g in train_dset.mg_cache._mgs])
    v_mean, v_std = train_V.mean(0), train_V.std(0) + 1e-6
    e_mean, e_std = train_E.mean(0), train_E.std(0) + 1e-6

    for dset in all_dsets:
        dset.mg_cache._mgs = [
            g._replace(V=(g.V - v_mean) / v_std, E=(g.E - e_mean) / e_std)
            for g in dset.mg_cache._mgs
        ]


# --------------------------------------------------------------------------- #
# 3. Model + training                                                          #
# --------------------------------------------------------------------------- #
def build_model(atom_fdim: int, bond_fdim: int, hidden: int, depth: int) -> MPNN:
    return MPNN(
        message_passing=BondMessagePassing(d_v=atom_fdim, d_e=bond_fdim, d_h=hidden, depth=depth),
        agg=MeanAggregation(d_h=hidden),
        predictor=RegressionFFN(input_dim=hidden, n_tasks=1, hidden_dim=hidden, n_layers=2),
        batch_norm=True,
    )


@torch.no_grad()
def predict(dset: MoleculeDataset, model: MPNN, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = build_dataloader(dset, batch_size=batch_size, shuffle=False, drop_last=False)
    preds = []
    for batch in loader:
        bmg = batch.bmg
        bmg.to(DEVICE)  # moves in place and returns None
        preds.append(model(bmg, batch.V_d, batch.X_d).cpu().numpy())
    return np.concatenate(preds, axis=0)


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def train_model(
    train_dset: MoleculeDataset,
    val_dset: MoleculeDataset,
    atom_fdim: int,
    bond_fdim: int,
    hidden: int,
    depth: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> MPNN:
    model = build_model(atom_fdim, bond_fdim, hidden, depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    train_loader = build_dataloader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val, best_state, patience, bad = float("inf"), None, 8, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            bmg, y = batch.bmg, batch.Y.to(DEVICE)
            bmg.to(DEVICE)  # moves in place and returns None
            opt.zero_grad()
            loss = F.mse_loss(model(bmg, batch.V_d, batch.X_d), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_loss += loss.item()

        val_pred = predict(val_dset, model, batch_size)
        val_rmse = rmse(val_pred, val_dset.Y.numpy() if isinstance(val_dset.Y, torch.Tensor) else val_dset.Y)
        sched.step(val_rmse)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | train MSE {run_loss/len(train_loader):.4f} | val RMSE {val_rmse:.4f}")

        if val_rmse < best_val - 1e-4:
            best_val, bad = val_rmse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stopping at epoch {epoch} (best val RMSE {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"  training done in {time.time()-t0:.1f}s | best val RMSE {best_val:.4f}")
    return model


# --------------------------------------------------------------------------- #
# 4. Main                                                                      #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="BONAFIDE + chemprop ESOL demo")
    parser.add_argument(
        "csv",
        nargs="?",
        default="/home/jackson/interpretable_aqueous_solubility/data/esol.csv",
        help="path to an ESOL-style CSV with SMILES and logS columns",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device: {DEVICE}\n")

    # --- Pick the BONAFIDE feature set (RDKit 2D numeric: fast, no external programs) ---
    atom_idx = bonafide_feature_indices(
        "atom", origin="rdkit", dimensionality="2D", requires_electronic_structure_data=False
    )
    bond_idx = bonafide_feature_indices(
        "bond", origin="rdkit", dimensionality="2D", requires_electronic_structure_data=False
    )
    print(f"BONAFIDE features: {len(atom_idx)} atom + {len(bond_idx)} bond (RDKit 2D numeric)\n")

    # --- Build the featurizer (BONAFIDE-only: no base multi-hot graph) ---
    featurizer = BonafideMolGraphFeaturizer(
        bonafide_atom_indices=atom_idx, bonafide_bond_indices=bond_idx
    )
    print(f"featurizer shape (atom_fdim, bond_fdim) = {featurizer.shape}\n")

    # --- Load + split the benchmark ---
    smiles, targets = load_esol(args.csv)
    train, val, test = random_split(smiles, targets, seed=args.seed)
    print(f"ESOL: {len(smiles)} molecules -> train {len(train.smiles)} / val {len(val.smiles)} / test {len(test.smiles)}\n")

    # --- Standardize the target with TRAIN-only statistics (as chemprop does) ---
    t_mean, t_std = float(train.targets.mean()), float(train.targets.std()) + 1e-6
    for s in (train, val, test):
        s.targets = (s.targets - t_mean) / t_std

    # --- Featurize (parallel) ---
    print("featurizing (BONAFIDE, parallel) ...")
    t0 = time.time()
    dsets = [to_dataset(s, featurizer, n_workers=4) for s in (train, val, test)]
    print(f"  done in {time.time()-t0:.1f}s\n")

    # --- Standardize node/edge features with train-only statistics ---
    standardize_features(dsets[0], dsets)

    # --- Train ---
    print("training MPNN ...")
    model = train_model(
        dsets[0], dsets[1],
        atom_fdim=featurizer.atom_fdim, bond_fdim=featurizer.bond_fdim,
        hidden=args.hidden, depth=args.depth, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
    )

    # --- Evaluate on all splits (inverse-transform predictions to original logS units) ---
    print("\n" + "=" * 64)
    print("RESULTS  (BONAFIDE-only features, MPNN on ESOL logS)")
    print("=" * 64)
    for name, dset in zip(("train", "val", "test"), dsets):
        y_std = dset.Y.numpy() if isinstance(dset.Y, torch.Tensor) else dset.Y
        pred_std = predict(dset, model, batch_size=256)
        # back to original units
        y_orig = y_std * t_std + t_mean
        pred_orig = pred_std * t_std + t_mean
        r = pearsonr(pred_orig.ravel(), y_orig.ravel())[0]
        print(f"  {name:5s}: RMSE {rmse(pred_orig, y_orig):.4f}   Pearson R {r:.4f}   (n={len(y_orig)})")
    # train-mean baseline in original units
    y_te = (dsets[2].Y.numpy() if isinstance(dsets[2].Y, torch.Tensor) else dsets[2].Y) * t_std + t_mean
    base_orig = t_mean  # predicting the train mean == 0 in standardized space
    print(f"  {'base':5s}: RMSE {rmse(np.full_like(y_te, base_orig), y_te):.4f}   (train-mean baseline)")
    print("=" * 64)
    print("\nDone. The model used ONLY BONAFIDE atom/bond features (no multi-hot graph).")


if __name__ == "__main__":
    main()
