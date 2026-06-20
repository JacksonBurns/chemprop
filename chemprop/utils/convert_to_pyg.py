"""Convert trained Chemprop models to PyTorch Geometric (PyG) models.

This module provides utilities to convert a Chemprop MPNN model checkpoint into an
equivalent PyG model that can be used for inference and training within the
``torch_geometric`` framework.

Usage
-----
CLI:

.. code-block:: bash

    chemprop convert --conversion to_pyg -i model.pt -o model_pyg.pt

Python:

.. code-block:: python

    from chemprop.utils.convert_to_pyg import convert_model_to_pyg

    pyg_model = convert_model_to_pyg("model.pt")
    # pyg_model is a PyG nn.Module that accepts torch_geometric.data.Batch
"""

from __future__ import annotations

import logging
from os import PathLike
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# Activation string to nn.Module mapping (mirrors chemprop.nn.utils)
ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "RELU": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "LEAKYRELU": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "PRELU": nn.PReLU,
    "tanh": nn.Tanh,
    "TANH": nn.Tanh,
    "elu": nn.ELU,
    "ELU": nn.ELU,
}


def _build_activation(activation: Any) -> nn.Module:
    """Build an activation module from a Chemprop activation spec."""
    if isinstance(activation, nn.Module):
        return activation
    if isinstance(activation, str):
        cls = ACTIVATION_MAP.get(activation)
        if cls is None:
            raise ValueError(f"Unsupported activation: {activation}")
        if cls is nn.LeakyReLU:
            return cls(negative_slope=0.1)
        return cls()
    raise ValueError(f"Unsupported activation type: {type(activation)}")


def _get_agg_type(hparams: dict) -> str:
    """Get aggregation type string from hparams."""
    cls = hparams.get("cls")
    if cls is None:
        return "mean"
    name = cls.__name__.lower()
    if "mean" in name:
        return "mean"
    if "sum" in name and "norm" not in name:
        return "sum"
    if "norm" in name:
        return "norm"
    if "attentive" in name:
        return "attentive"
    return "mean"


class ChempropBondMessagePassingPyG(nn.Module):
    """PyG-compatible bond-based message passing, equivalent to Chemprop's BondMessagePassing.

    This module implements the same message passing algorithm as Chemprop but using
    standard PyTorch operations compatible with ``torch_geometric`` data objects.
    """

    def __init__(
        self,
        d_v: int,
        d_e: int,
        d_h: int,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        undirected: bool = False,
        d_vd: int | None = None,
        bias: bool = False,
        V_transform: nn.Module | None = None,
        E_transform: nn.Module | None = None,
        V_d_transform: nn.Module | None = None,
    ):
        super().__init__()
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.depth = depth
        self.undirected = undirected
        self.d_vd = d_vd

        self.V_transform = V_transform or nn.Identity()
        self.E_transform = E_transform or nn.Identity()
        self.V_d_transform = V_d_transform or nn.Identity()

        self.W_i = nn.Linear(d_v + d_e, d_h, bias=bias)
        self.W_h = nn.Linear(d_h, d_h, bias=bias)
        self.W_o = nn.Linear(d_v + d_h, d_h)
        self.W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        self.dropout = nn.Dropout(dropout)
        self.tau = _build_activation(activation)

    @property
    def output_dim(self) -> int:
        return self.W_d.out_features if self.W_d is not None else self.W_o.out_features

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        rev_edge_index: Tensor,
        batch: Tensor,
        v_d: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run bond-based message passing.

        Parameters
        ----------
        x : Tensor
            Node features, shape (num_nodes, d_v).
        edge_index : Tensor
            Edge connectivity, shape (2, num_edges).
        edge_attr : Tensor
            Edge features, shape (num_edges, d_e).
        rev_edge_index : Tensor
            Reverse edge index mapping, shape (num_edges,).
        batch : Tensor
            Graph assignment per node, shape (num_nodes,).
        v_d : Tensor | None
            Optional vertex descriptors, shape (num_nodes, d_vd).

        Returns
        -------
        node_h : Tensor
            Final node embeddings, shape (num_nodes, output_dim).
        graph_h : Tensor
            Aggregated graph embeddings, shape (num_graphs, output_dim).
        """
        src = edge_index[0]
        dst = edge_index[1]

        # Apply GraphTransform
        x = self.V_transform(x)
        edge_attr = self.E_transform(edge_attr)

        # Initialize: h_ew = tau(W_i([v_s; e]))
        src_feat = x[src]
        h_0 = self.W_i(torch.cat([src_feat, edge_attr], dim=1))

        h = self.tau(h_0)

        for _ in range(1, self.depth):
            if self.undirected:
                h = (h + h[rev_edge_index]) / 2

            # Message: M_vw = sum_u h_uw - h_rev
            index_torch = dst.unsqueeze(1).expand(-1, h.shape[1])
            m_all = torch.zeros(
                len(x), h.shape[1], dtype=h.dtype, device=h.device
            ).scatter_reduce_(0, index_torch, h, reduce="sum", include_self=False)
            m_all = m_all[src]
            m_rev = h[rev_edge_index]
            m = m_all - m_rev

            h = self.tau(h_0 + self.W_h(m))
            h = self.dropout(h)

        # Aggregate messages to vertices
        index_torch = dst.unsqueeze(1).expand(-1, h.shape[1])
        m_v = torch.zeros(
            len(x), h.shape[1], dtype=h.dtype, device=h.device
        ).scatter_reduce_(0, index_torch, h, reduce="sum", include_self=False)

        # Finalize
        node_h = self.W_o(torch.cat([x, m_v], dim=1))
        node_h = self.tau(node_h)
        node_h = self.dropout(node_h)

        if v_d is not None:
            v_d = self.V_d_transform(v_d)
            node_h = self.W_d(torch.cat([node_h, v_d], dim=1))
            node_h = self.dropout(node_h)

        return node_h, node_h


class ChempropAtomMessagePassingPyG(nn.Module):
    """PyG-compatible atom-based message passing, equivalent to Chemprop's AtomMessagePassing.

    This module implements the same message passing algorithm as Chemprop but using
    standard PyTorch operations compatible with ``torch_geometric`` data objects.
    """

    def __init__(
        self,
        d_v: int,
        d_e: int,
        d_h: int,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        undirected: bool = False,
        d_vd: int | None = None,
        bias: bool = False,
        V_transform: nn.Module | None = None,
        E_transform: nn.Module | None = None,
        V_d_transform: nn.Module | None = None,
    ):
        super().__init__()
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.depth = depth
        self.undirected = undirected
        self.d_vd = d_vd

        self.V_transform = V_transform or nn.Identity()
        self.E_transform = E_transform or nn.Identity()
        self.V_d_transform = V_d_transform or nn.Identity()

        self.W_i = nn.Linear(d_v, d_h, bias=bias)
        self.W_h = nn.Linear(d_e + d_h, d_h, bias=bias)
        self.W_o = nn.Linear(d_v + d_h, d_h)
        self.W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        self.dropout = nn.Dropout(dropout)
        self.tau = _build_activation(activation)

    @property
    def output_dim(self) -> int:
        return self.W_d.out_features if self.W_d is not None else self.W_o.out_features

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        rev_edge_index: Tensor,
        batch: Tensor,
        v_d: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run atom-based message passing.

        Parameters
        ----------
        x : Tensor
            Node features, shape (num_nodes, d_v).
        edge_index : Tensor
            Edge connectivity, shape (2, num_edges).
        edge_attr : Tensor
            Edge features, shape (num_edges, d_e).
        rev_edge_index : Tensor
            Reverse edge index mapping, shape (num_edges,).
        batch : Tensor
            Graph assignment per node, shape (num_nodes,).
        v_d : Tensor | None
            Optional vertex descriptors, shape (num_nodes, d_vd).

        Returns
        -------
        node_h : Tensor
            Final node embeddings, shape (num_nodes, output_dim).
        graph_h : Tensor
            Aggregated graph embeddings, shape (num_graphs, output_dim).
        """
        src = edge_index[0]
        dst = edge_index[1]

        # Apply GraphTransform
        x = self.V_transform(x)
        edge_attr = self.E_transform(edge_attr)

        # Initialize: h_v = tau(W_i(v))
        h_0 = self.W_i(x[src])

        h = self.tau(h_0)

        for _ in range(1, self.depth):
            if self.undirected:
                h = (h + h[rev_edge_index]) / 2

            # Message: M_v = sum_u [h_u; e_uv]
            h_with_e = torch.cat([h, edge_attr], dim=1)
            index_torch = dst.unsqueeze(1).expand(-1, h_with_e.shape[1])
            m = torch.zeros(
                len(x), h_with_e.shape[1], dtype=h_with_e.dtype, device=h_with_e.device
            ).scatter_reduce_(
                0, index_torch, h_with_e, reduce="sum", include_self=False
            )[src]

            h = self.tau(h_0 + self.W_h(m))
            h = self.dropout(h)

        # Aggregate messages to vertices
        index_torch = dst.unsqueeze(1).expand(-1, h.shape[1])
        m_v = torch.zeros(
            len(x), h.shape[1], dtype=h.dtype, device=h.device
        ).scatter_reduce_(0, index_torch, h, reduce="sum", include_self=False)

        # Finalize
        node_h = self.W_o(torch.cat([x, m_v], dim=1))
        node_h = self.tau(node_h)
        node_h = self.dropout(node_h)

        if v_d is not None:
            v_d = self.V_d_transform(v_d)
            node_h = self.W_d(torch.cat([node_h, v_d], dim=1))
            node_h = self.dropout(node_h)

        return node_h, node_h


class ChempropPredictorPyG(nn.Module):
    """PyG-compatible predictor FFN, equivalent to Chemprop's predictor modules.

    Supports regression, binary classification, and multiclass classification.
    """

    def __init__(
        self,
        predictor_hparams: dict,
        predictor_state_dict: dict,
    ):
        super().__init__()
        cls_name = predictor_hparams.get("cls", __class__).__name__
        self.predictor_type = cls_name.lower()

        input_dim = predictor_hparams.get("input_dim", 300)
        n_tasks = predictor_hparams.get("n_tasks", 1)
        hidden_dim = predictor_hparams.get("hidden_dim", 300)
        n_layers = predictor_hparams.get("n_layers", 1)
        dropout = predictor_hparams.get("dropout", 0.0)
        activation = predictor_hparams.get("activation", "relu")
        n_classes = predictor_hparams.get("n_classes", None)

        if n_classes is not None:
            output_dim = n_tasks * n_classes
        else:
            if "mve" in cls_name.lower() or "quantile" in cls_name.lower():
                output_dim = n_tasks * 2
            elif "evidential" in cls_name.lower():
                output_dim = n_tasks * 4
            else:
                output_dim = n_tasks

        # Build FFN
        self.ffn = self._build_mlp(
            input_dim, output_dim, hidden_dim, n_layers, dropout, activation
        )

        # Load FFN weights from state dict
        ffn_state = {
            k.replace("ffn.", ""): v
            for k, v in predictor_state_dict.items()
            if k.startswith("ffn.")
        }
        self.ffn.load_state_dict(ffn_state, strict=False)

        # Output transform (for normalized regression)
        self.output_transform = self._build_output_transform(predictor_state_dict)

        # n_classes for multiclass
        self.n_classes = n_classes

        # Spectral activation
        self.spectral_activation = None
        if "spectral" in self.predictor_type:
            spectral_act_name = predictor_hparams.get("spectral_activation", "softplus")
            if spectral_act_name == "exp":
                self.spectral_activation = _Exp()
            else:
                self.spectral_activation = nn.Softplus()

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | list[int],
        n_layers: int,
        dropout: float,
        activation: str,
    ) -> nn.Sequential:
        """Build an MLP matching Chemprop's nested Sequential block structure.

        Chemprop's MLP.build() produces:
          Sequential(
            Sequential(Linear(in, h1)),
            Sequential(act, dropout, Linear(h1, h2)),
            ...
            Sequential(act, dropout, Linear(hN, out)),
          )

        This matches state dict keys like:
          0.0.weight, 0.0.bias  (first Linear)
          1.2.weight, 1.2.bias  (second Linear, index 2 in its Sequential)
        """
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * n_layers
        else:
            hidden_dims = list(hidden_dim)

        dims = [input_dim] + hidden_dims + [output_dim]

        # Block 0: Sequential(Linear(input_dim, hidden_dim[0]))
        blocks = [nn.Sequential(nn.Linear(dims[0], dims[1]))]

        # Block 1+: Sequential(activation, dropout, Linear(dims[i], dims[i+1]))
        if len(dims) > 2:
            blocks.extend(
                [
                    nn.Sequential(
                        _build_activation(activation),
                        nn.Dropout(dropout),
                        nn.Linear(d1, d2),
                    )
                    for d1, d2 in zip(dims[1:-1], dims[2:])
                ]
            )

        return nn.Sequential(*blocks)

    def _build_output_transform(self, state_dict: dict) -> nn.Module:
        """Build output transform from state dict if present."""
        has_transform = any(k.startswith("output_transform.") for k in state_dict)
        if not has_transform:
            return nn.Identity()

        mean = state_dict.get("output_transform.mean")
        scale = state_dict.get("output_transform.scale")
        if mean is not None and scale is not None:
            return _UnscaleTransform(mean.clone(), scale.clone())
        return nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through predictor."""
        y = self.ffn(x)

        if "spectral" in self.predictor_type:
            y = self.spectral_activation(y)
            return y / y.sum(1, keepdim=True)

        if "binaryclassificationffn" in self.predictor_type:
            return y.sigmoid()

        if "multiclassclassificationffn" in self.predictor_type and self.n_classes:
            return y.reshape(y.shape[0], -1, self.n_classes).softmax(-1)

        if "binarydirichlet" in self.predictor_type:
            y = y.reshape(y.shape[0], -1, 2)
            alpha = torch.nn.functional.softplus(y) + 1
            y = alpha / alpha.sum(-1, keepdim=True)
            u = 2 / alpha.sum(-1)
            return torch.stack([y[..., 1], u], dim=2)

        if "multiclassdirichlet" in self.predictor_type and self.n_classes:
            y = y.reshape(y.shape[0], -1, self.n_classes)
            alpha = torch.nn.functional.softplus(y) + 1
            return alpha / alpha.sum(-1, keepdim=True)

        if "mve" in self.predictor_type:
            mean, var = torch.chunk(y, 2, dim=1)
            var = torch.nn.functional.softplus(var)
            mean = self.output_transform(mean)
            if not isinstance(self.output_transform, nn.Identity):
                var = var * (self.output_transform.scale ** 2)
            return torch.stack([mean, var], dim=2)

        if "evidential" in self.predictor_type:
            mean, v, alpha, beta = torch.chunk(y, 4, dim=1)
            v = torch.nn.functional.softplus(v)
            alpha = torch.nn.functional.softplus(alpha) + 1
            beta = torch.nn.functional.softplus(beta)
            mean = self.output_transform(mean)
            if not isinstance(self.output_transform, nn.Identity):
                beta = beta * (self.output_transform.scale ** 2)
            return torch.stack([mean, v, alpha, beta], dim=2)

        if "quantile" in self.predictor_type:
            lower, upper = torch.chunk(y, 2, dim=1)
            lower = self.output_transform(lower)
            upper = self.output_transform(upper)
            mean = (lower + upper) / 2
            interval = upper - lower
            return torch.stack([mean, interval], dim=2)

        # Standard regression
        return self.output_transform(y)


class _UnscaleTransform(nn.Module):
    """Replicates Chemprop's UnscaleTransform."""

    def __init__(self, mean: Tensor, scale: Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x
        return x * self.scale + self.mean


class _ScaleTransform(nn.Module):
    """Replicates Chemprop's ScaleTransform: (x - mean) / scale in eval mode, identity in train."""

    def __init__(self, mean: Tensor, scale: Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x
        return (x - self.mean) / self.scale


class _Exp(nn.Module):
    """Element-wise exponential for spectral activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x.exp()


class ChempropMPNNPyG(nn.Module):
    """Full PyG-compatible Chemprop MPNN model.

    This model replicates the full Chemprop pipeline:
    message_passing -> aggregation -> batch_norm -> predictor

    It accepts standard ``torch_geometric.data.Data`` or
    ``torch_geometric.data.Batch`` objects.

    Parameters
    ----------
    message_passing : nn.Module
        The message passing module (ChempropBondMessagePassingPyG or
        ChempropAtomMessagePassingPyG).
    agg_type : str
        Aggregation type: 'mean', 'sum', 'norm', or 'attentive'.
    norm : float
        Normalization constant for 'norm' aggregation.
    predictor : ChempropPredictorPyG
        The predictor FFN module.
    batch_norm : bool
        Whether to apply batch normalization before the predictor.

    Example
    -------
    >>> model = ChempropMPNNPyG(...)
    >>> # For a single molecule:
    >>> pred = model(data.x, data.edge_index, data.edge_attr,
    ...              data.rev_edge_index, data.batch)
    >>> # For a batch of molecules:
    >>> preds = model(batch.x, batch.edge_index, batch.edge_attr,
    ...               batch.rev_edge_index, batch.batch)
    """

    def __init__(
        self,
        message_passing: nn.Module,
        agg_type: str = "mean",
        norm: float = 100.0,
        predictor: ChempropPredictorPyG | None = None,
        batch_norm: bool = False,
        X_d_transform: nn.Module | None = None,
        agg_attention: nn.Module | None = None,
    ):
        super().__init__()
        self.message_passing = message_passing
        self.agg_type = agg_type
        self.norm = norm
        self.X_d_transform = X_d_transform or nn.Identity()
        self.agg_attention = agg_attention

        if batch_norm:
            self.bn = nn.BatchNorm1d(self.message_passing.output_dim)
        else:
            self.bn = nn.Identity()

        self.predictor = predictor

    def aggregate(self, node_h: Tensor, batch: Tensor) -> Tensor:
        """Aggregate node embeddings to graph-level embeddings."""
        num_graphs = int(batch.max().item()) + 1 if len(batch) > 0 else 0
        if num_graphs == 0:
            return torch.zeros(0, node_h.shape[1], dtype=node_h.dtype, device=node_h.device)

        if self.agg_type == "mean":
            graph_h = torch.zeros(
                num_graphs, node_h.shape[1], dtype=node_h.dtype, device=node_h.device
            ).scatter_reduce_(
                0,
                batch.unsqueeze(1).expand(-1, node_h.shape[1]),
                node_h,
                reduce="mean",
                include_self=False,
            )
        elif self.agg_type == "sum":
            graph_h = torch.zeros(
                num_graphs, node_h.shape[1], dtype=node_h.dtype, device=node_h.device
            ).scatter_reduce_(
                0,
                batch.unsqueeze(1).expand(-1, node_h.shape[1]),
                node_h,
                reduce="sum",
                include_self=False,
            )
        elif self.agg_type == "norm":
            graph_h = torch.zeros(
                num_graphs, node_h.shape[1], dtype=node_h.dtype, device=node_h.device
            ).scatter_reduce_(
                0,
                batch.unsqueeze(1).expand(-1, node_h.shape[1]),
                node_h,
                reduce="sum",
                include_self=False,
            )
            graph_h = graph_h / self.norm
        elif self.agg_type == "attentive":
            attention_logits = self.agg_attention(node_h).exp()
            Z = torch.zeros(
                num_graphs, 1, dtype=node_h.dtype, device=node_h.device
            ).scatter_reduce_(
                0, batch.unsqueeze(1), attention_logits, reduce="sum", include_self=False
            )
            alphas = attention_logits / Z[batch]
            index_torch = batch.unsqueeze(1).expand(-1, node_h.shape[1])
            graph_h = torch.zeros(
                num_graphs, node_h.shape[1], dtype=node_h.dtype, device=node_h.device
            ).scatter_reduce_(
                0, index_torch, alphas * node_h, reduce="sum", include_self=False
            )
        else:
            raise ValueError(f"Unsupported aggregation type: {self.agg_type}")

        return graph_h

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        rev_edge_index: Tensor,
        batch: Tensor,
        v_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Node features, shape (num_nodes, d_v).
        edge_index : Tensor
            Edge connectivity, shape (2, num_edges).
        edge_attr : Tensor
            Edge features, shape (num_edges, d_e).
        rev_edge_index : Tensor
            Reverse edge index mapping, shape (num_edges,).
        batch : Tensor
            Graph assignment per node, shape (num_nodes,).
        v_d : Tensor | None
            Optional vertex descriptors.
        X_d : Tensor | None
            Optional molecule-level descriptors.

        Returns
        -------
        Tensor
            Predictions, shape (num_graphs, output_dim).
        """
        node_h, _ = self.message_passing(x, edge_index, edge_attr, rev_edge_index, batch, v_d)
        graph_h = self.aggregate(node_h, batch)
        graph_h = self.bn(graph_h)
        if X_d is not None:
            X_d = self.X_d_transform(X_d)
            graph_h = torch.cat([graph_h, X_d], dim=1)
        return self.predictor(graph_h)

    @property
    def output_dim(self) -> int:
        return self.predictor.ffn[-1].out_features if hasattr(self.predictor, "ffn") else 1


def _is_bond_message_passing(mp_hparams: dict) -> bool:
    """Check if message passing is bond-based."""
    cls = mp_hparams.get("cls")
    if cls is None:
        return True
    return "bond" in cls.__name__.lower()


def convert_model_to_pyg(
    checkpoint_path: str | PathLike,
    map_location: str | torch.device | None = None,
) -> ChempropMPNNPyG:
    """Convert a Chemprop model checkpoint to a PyG-compatible model.

    This function loads a Chemprop model checkpoint and constructs an equivalent
    PyG model with the same architecture and weights. The returned model can be
    used for inference and training within the ``torch_geometric`` framework.

    Parameters
    ----------
    checkpoint_path : str or PathLike
        Path to the Chemprop model checkpoint (.pt file).
    map_location : str, torch.device, or None
        Device to load tensors on.

    Returns
    -------
    ChempropMPNNPyG
        A PyG-compatible model with the same architecture and weights.

    Raises
    ------
    ImportError
        If ``torch_geometric`` is not installed.
    RuntimeError
        If the checkpoint format is not supported.

    Example
    -------
    >>> model = convert_model_to_pyg("chemprop_model.pt")
    >>> # Use with PyG data:
    >>> pred = model(data.x, data.edge_index, data.edge_attr,
    ...              data.rev_edge_index, data.batch)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu", weights_only=False)

    hparams = checkpoint.get("hyper_parameters")
    state_dict = checkpoint.get("state_dict")

    if hparams is None or state_dict is None:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is missing 'hyper_parameters' or 'state_dict'. "
            "This may be a Lightning checkpoint; try converting with v2_0_to_v2_1 first."
        )

    # Extract message passing params
    mp_hparams = hparams.get("message_passing", {})
    mp_prefix = "message_passing."

    # Detect unsupported model types
    cls = mp_hparams.get("cls")
    cls_name = cls.__name__ if cls else ""

    if "multicomponent" in cls_name.lower():
        raise RuntimeError(
            "MulticomponentMPNN (multi-molecule) models are not supported for PyG conversion. "
            "Only single-component MPNN models can be converted."
        )

    if "mab" in cls_name.lower():
        raise RuntimeError(
            "MolAtomBondMPNN (atom/bond-level prediction) models are not supported for PyG conversion. "
            "Only molecule-level MPNN models can be converted."
        )

    if any(k in hparams for k in ("mol_predictor", "atom_predictor", "bond_predictor")):
        raise RuntimeError(
            "MolAtomBondMPNN models detected. These are not supported for PyG conversion. "
            "Only molecule-level MPNN models can be converted."
        )

    d_v = mp_hparams.get("d_v", 77)
    d_e = mp_hparams.get("d_e", 14)
    d_h = mp_hparams.get("d_h", 300)
    depth = mp_hparams.get("depth", 3)
    dropout = mp_hparams.get("dropout", 0.0)
    activation = mp_hparams.get("activation", "relu")
    undirected = mp_hparams.get("undirected", False)
    d_vd = mp_hparams.get("d_vd", None)
    bias = mp_hparams.get("bias", False)

    # Detect and construct transforms
    V_transform = None
    E_transform = None
    if any(k.startswith("message_passing.graph_transform.V_transform.") for k in state_dict):
        V_transform = _ScaleTransform(
            state_dict["message_passing.graph_transform.V_transform.mean"].clone(),
            state_dict["message_passing.graph_transform.V_transform.scale"].clone(),
        )
    if any(k.startswith("message_passing.graph_transform.E_transform.") for k in state_dict):
        E_transform = _ScaleTransform(
            state_dict["message_passing.graph_transform.E_transform.mean"].clone(),
            state_dict["message_passing.graph_transform.E_transform.scale"].clone(),
        )

    V_d_transform = None
    if any(k.startswith("message_passing.V_d_transform.") for k in state_dict):
        V_d_transform = _ScaleTransform(
            state_dict["message_passing.V_d_transform.mean"].clone(),
            state_dict["message_passing.V_d_transform.scale"].clone(),
        )

    X_d_transform = None
    if any(k.startswith("X_d_transform.") for k in state_dict):
        X_d_transform = _ScaleTransform(
            state_dict["X_d_transform.mean"].clone(),
            state_dict["X_d_transform.scale"].clone(),
        )

    # Build message passing module
    if _is_bond_message_passing(mp_hparams):
        mp = ChempropBondMessagePassingPyG(
            d_v=d_v,
            d_e=d_e,
            d_h=d_h,
            depth=depth,
            dropout=dropout,
            activation=activation,
            undirected=undirected,
            d_vd=d_vd,
            bias=bias,
            V_transform=V_transform,
            E_transform=E_transform,
            V_d_transform=V_d_transform,
        )
    else:
        mp = ChempropAtomMessagePassingPyG(
            d_v=d_v,
            d_e=d_e,
            d_h=d_h,
            depth=depth,
            dropout=dropout,
            activation=activation,
            undirected=undirected,
            d_vd=d_vd,
            bias=bias,
            V_transform=V_transform,
            E_transform=E_transform,
            V_d_transform=V_d_transform,
        )

    # Load message passing weights
    mp_state = {
        k.replace(mp_prefix, ""): v
        for k, v in state_dict.items()
        if k.startswith(mp_prefix)
    }
    mp.load_state_dict(mp_state, strict=False)

    # Extract aggregation params
    agg_hparams = hparams.get("agg", {})
    agg_type = _get_agg_type(agg_hparams)
    norm_val = agg_hparams.get("norm", 100.0)

    # Construct attentive aggregation weights if needed
    agg_attention = None
    if agg_type == "attentive":
        output_size = mp.output_dim
        agg_attention = nn.Linear(output_size, 1)
        agg_attention.load_state_dict(
            {
                "weight": state_dict["agg.W.weight"].clone(),
                "bias": state_dict["agg.W.bias"].clone(),
            }
        )

    # Extract predictor params
    pred_hparams = hparams.get("predictor", {})
    pred_prefix = "predictor."
    pred_state = {
        k.replace(pred_prefix, ""): v
        for k, v in state_dict.items()
        if k.startswith(pred_prefix)
    }
    predictor = ChempropPredictorPyG(pred_hparams, pred_state)

    # Extract batch norm
    batch_norm = hparams.get("batch_norm", False)
    model = ChempropMPNNPyG(
        message_passing=mp,
        agg_type=agg_type,
        norm=norm_val,
        predictor=predictor,
        batch_norm=batch_norm,
        X_d_transform=X_d_transform,
        agg_attention=agg_attention,
    )

    # Load batch norm weights if present
    if batch_norm:
        bn_state = {
            k.replace("bn.", ""): v
            for k, v in state_dict.items()
            if k.startswith("bn.")
        }
        if bn_state:
            model.bn.load_state_dict(bn_state, strict=False)

    return model


def convert_model_file_to_pyg(
    input_path: str | PathLike,
    output_path: str | PathLike,
) -> None:
    """Convert a Chemprop model checkpoint to a PyG model file.

    The output file contains a PyG-compatible model that can be loaded and used
    independently of Chemprop (though the conversion itself requires Chemprop).

    Parameters
    ----------
    input_path : str or PathLike
        Path to the Chemprop model checkpoint (.pt file).
    output_path : str or PathLike
        Path to save the converted PyG model (.pt file).
    """
    logger.info(f"Converting Chemprop model '{input_path}' to PyG format...")

    model = convert_model_to_pyg(input_path)

    pyg_checkpoint = {
        "hyper_parameters": {
            "message_passing_type": "bond"
            if isinstance(model.message_passing, ChempropBondMessagePassingPyG)
            else "atom",
            "agg_type": model.agg_type,
            "norm": model.norm,
            "batch_norm": not isinstance(model.bn, nn.Identity),
            "input_dim": model.message_passing.d_v,
            "edge_dim": model.message_passing.d_e,
            "hidden_dim": model.message_passing.d_h,
            "output_dim": model.predictor.ffn[-1][-1].out_features,
        },
        "state_dict": model.state_dict(),
    }

    torch.save(pyg_checkpoint, output_path)
    logger.info(f"PyG model saved to '{output_path}'")


def load_pyg_model(
    checkpoint_path: str | PathLike,
    map_location: str | torch.device | None = None,
) -> ChempropMPNNPyG:
    """Load a PyG-converted Chemprop model.

    Parameters
    ----------
    checkpoint_path : str or PathLike
        Path to the PyG model checkpoint file.
    map_location : str, torch.device, or None
        Device to load tensors on.

    Returns
    -------
    ChempropMPNNPyG
        The loaded PyG-compatible model.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location or "cpu", weights_only=False
    )

    hparams = checkpoint["hyper_parameters"]
    mp_type = hparams.get("message_passing_type", "bond")

    d_v = hparams.get("input_dim", 77)
    d_e = hparams.get("edge_dim", 14)
    d_h = hparams.get("hidden_dim", 300)

    # Reconstruct the message passing module
    # We need to extract parameters from the state dict
    state_dict = checkpoint["state_dict"]

    if mp_type == "bond":
        mp = ChempropBondMessagePassingPyG(d_v=d_v, d_e=d_e, d_h=d_h)
    else:
        mp = ChempropAtomMessagePassingPyG(d_v=d_v, d_e=d_e, d_h=d_h)

    # Load message passing state
    mp_state = {k: v for k, v in state_dict.items() if k.startswith("message_passing.")}
    mp.load_state_dict(mp_state, strict=False)

    # Build predictor - extract dimensions from state dict
    pred_state = {k: v for k, v in state_dict.items() if k.startswith("predictor.")}

    # Extract FFN dimensions from state dict
    ffn_w0 = pred_state.get("ffn.0.0.weight")
    ffn_w1 = pred_state.get("ffn.1.2.weight")

    if ffn_w0 is not None:
        pred_input_dim = ffn_w0.shape[1]
        pred_hidden_dim = ffn_w0.shape[0]
        pred_output_dim = ffn_w1.shape[0] if ffn_w1 is not None else pred_hidden_dim
    else:
        pred_input_dim = d_h
        pred_hidden_dim = d_h
        pred_output_dim = 1

    # Count FFN layers from state dict keys
    ffn_keys = [k for k in pred_state.keys() if k.startswith("ffn.")]
    block_indices = set()
    for k in ffn_keys:
        parts = k.split(".")
        if len(parts) >= 2:
            try:
                block_indices.add(int(parts[1]))
            except ValueError:
                pass
    n_layers = max(block_indices) if block_indices else 1

    # Detect activation from PReLU weights
    has_prelu = any("ffn.{}.0.weight".format(i) in pred_state for i in range(1, 10))
    activation = "prelu" if has_prelu else "relu"

    predictor = ChempropPredictorPyG(
        {
            "input_dim": pred_input_dim,
            "hidden_dim": pred_hidden_dim,
            "n_tasks": pred_output_dim,
            "n_layers": n_layers,
            "dropout": 0.0,
            "activation": activation,
        },
        {},
    )
    predictor.load_state_dict(pred_state, strict=False)

    # Detect X_d_transform from state dict
    X_d_transform = None
    if "X_d_transform.mean" in state_dict and "X_d_transform.scale" in state_dict:
        X_d_transform = _ScaleTransform(
            state_dict["X_d_transform.mean"].clone(),
            state_dict["X_d_transform.scale"].clone(),
        )

    # Detect agg_attention from state dict
    agg_attention = None
    if "agg_attention.weight" in state_dict:
        w_shape = state_dict["agg_attention.weight"].shape
        agg_attention = nn.Linear(w_shape[1], w_shape[0])

    model = ChempropMPNNPyG(
        message_passing=mp,
        agg_type=hparams.get("agg_type", "mean"),
        norm=hparams.get("norm", 100.0),
        predictor=predictor,
        batch_norm=hparams.get("batch_norm", False),
        X_d_transform=X_d_transform,
        agg_attention=agg_attention,
    )

    model.load_state_dict(state_dict, strict=False)
    return model
