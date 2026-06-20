.. _convert:

Conversion
----------

Chemprop supports several model conversion workflows:

Chemprop v1 to v2
~~~~~~~~~~~~~~~~~

To convert a trained model from Chemprop v1 to v2, run ``chemprop convert`` and specify:

 * :code:`--input-path <path>` Path of the Chemprop v1 file to convert.
 * :code:`--output-path <path>` Path where the converted Chemprop v2 model will be saved. If unspecified, this will default to ``<CURRENT_DIRECTORY/STEM_OF_INPUT>_v2.pt``.

Chemprop v2.0 to v2.1
~~~~~~~~~~~~~~~~~~~~~

To convert a trained model from Chemprop v2.0.x to v2.1.y (or newer), run ``chemprop convert --conversion v2_0_to_v2_1`` and additionally specify:

 * :code:`--input-path <path>` Path of the Chemprop v2.0 file to convert.
 * :code:`--output-path <path>` Path where the converted Chemprop v2.1 model will be saved. If unspecified, this will default to ``<CURRENT_DIRECTORY/STEM_OF_INPUT>_v2_1.pt``.

Chemprop v2 to PyTorch Geometric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can convert a trained Chemprop v2 model into a PyTorch Geometric (PyG) model, enabling inference and training within the ``torch_geometric`` framework.

First, install the optional PyG dependency:

.. code-block::

    pip install chemprop[pyg]

Then convert your model:

.. code-block::

    chemprop convert --conversion to_pyg --input-path chemprop_model.pt --output-path pyg_model.pt

If ``--output-path`` is omitted, the file is saved as ``<STEM_OF_INPUT>_pyg.pt`` in the current directory.

The converted model can be loaded and used with standard PyG data objects:

.. code-block:: python

    from chemprop.utils.convert_to_pyg import load_pyg_model
    from torch_geometric.loader import DataLoader

    model = load_pyg_model("pyg_model.pt")
    model.eval()

    with torch.no_grad():
        preds = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.rev_edge_index,
            batch.batch,
        )

For more details, including the Python API, see :doc:`../python/pyg_conversion`.

