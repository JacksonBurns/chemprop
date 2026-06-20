.. _pyg_conversion:

Converting to PyTorch Geometric
===============================

Chemprop models can be converted to PyTorch Geometric (PyG) format, enabling you to use trained Chemprop models for inference and training within the ``torch_geometric`` framework.

Overview
--------

The conversion process takes a Chemprop v2 model checkpoint (``.pt`` file) and produces a PyG-compatible model with identical architecture and weights. The converted model accepts standard PyG data objects and produces the same predictions as the original Chemprop model.

Supported configurations:

* Bond-based and atom-based message passing
* Mean, sum, and norm aggregation
* Batch normalization
* Regression, binary classification, and multiclass classification
* Multi-task prediction
* Undirected message passing
* Multiple activation functions (ReLU, LeakyReLU, PReLU, Tanh, ELU)
* Arbitrary message passing depth and FFN layers

Installation
------------

Install the optional PyG dependency:

.. code-block::

    pip install chemprop[pyg]

CLI Usage
---------

Convert a model from the command line:

.. code-block::

    chemprop convert --conversion to_pyg --input-path chemprop_model.pt --output-path pyg_model.pt

The output file defaults to ``<input_stem>_pyg.pt`` in the current directory if ``--output-path`` is omitted.

Python API
----------

Convert a model programmatically:

.. code-block:: python

    from chemprop.utils.convert_to_pyg import convert_model_to_pyg

    pyg_model = convert_model_to_pyg("chemprop_model.pt")
    pyg_model.eval()

The returned model is a PyTorch ``nn.Module`` that accepts raw tensors:

.. code-block:: python

    with torch.no_grad():
        predictions = pyg_model(
            x=batch.x,             # node features: (num_nodes, d_v)
            edge_index=batch.edge_index,  # connectivity: (2, num_edges)
            edge_attr=batch.edge_attr,    # edge features: (num_edges, d_e)
            rev_edge_index=batch.rev_edge_index,  # reverse edge mapping: (num_edges,)
            batch=batch.batch,             # graph assignment: (num_nodes,)
        )

Model Architecture
------------------

The converted model replicates the full Chemprop pipeline:

.. code-block::

    node_features ──→ message_passing ──→ aggregation ──→ batch_norm ──→ predictor ──→ predictions

The message passing module is implemented as either ``ChempropBondMessagePassingPyG`` or ``ChempropAtomMessagePassingPyG``, depending on the original Chemprop model. These modules implement the same message passing algorithm as their Chemprop counterparts using standard PyTorch ``scatter_reduce_`` operations.

Loading a Converted Model
-------------------------

Converted models can be loaded independently of the Chemprop training pipeline:

.. code-block:: python

    from chemprop.utils.convert_to_pyg import load_pyg_model

    model = load_pyg_model("pyg_model.pt")
    model.eval()

Using with PyG Data Objects
----------------------------

The converted model works with ``torch_geometric.data.Data`` and ``torch_geometric.data.Batch`` objects. Note that PyG's default ``Data`` objects do not include the ``rev_edge_index`` attribute used by Chemprop's message passing. You will need to construct this attribute from your graph data.

.. code-block:: python

    from torch_geometric.data import Data, Batch

    # For a single molecule, construct rev_edge_index from directed edges.
    # Chemprop creates two directed edges per bond (u→v and v→u), and
    # rev_edge_index maps each directed edge to its reverse.
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        rev_edge_index=rev_edge_index,
    )

    # For a batch of molecules:
    batch = Batch.from_data_list([data1, data2, ...])
    preds = model(batch.x, batch.edge_index, batch.edge_attr,
                  batch.rev_edge_index, batch.batch)

Training a Converted Model
--------------------------

The converted model is a standard PyTorch module and can be trained using any optimizer and loss function:

.. code-block:: python

    import torch
    from torch.nn import MSELoss

    model = load_pyg_model("pyg_model.pt")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()

    for data in dataloader:
        optimizer.zero_grad()
        preds = model(data.x, data.edge_index, data.edge_attr,
                      data.rev_edge_index, data.batch)
        loss = criterion(preds, data.y)
        loss.backward()
        optimizer.step()
