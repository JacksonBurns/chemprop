from abc import abstractmethod
from torch import Tensor, nn, scatter_reduce

from chemprop.utils import ClassRegistry
from chemprop.nn.hparams import HasHParams


__all__ = [
    "Aggregation",
    "AggregationRegistry",
    "MeanAggregation",
    "SumAggregation",
    "NormAggregation",
    "AttentiveAggregation",
]


def _broadcast(src: Tensor, other: Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def _scatter_softmax(src: Tensor, index: Tensor, dim: int = -1) -> Tensor:
    index = _broadcast(index, src, dim)

    max_value_per_index = scatter_reduce(src, dim, index, src, "amax")[0]

    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = scatter_reduce(recentered_scores_exp, dim, index, recentered_scores_exp, "sum")
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)


class Aggregation(nn.Module, HasHParams):
    """An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
    a batch of graph-level representations

    .. note::
        this class is abstract and cannot be instantiated.

    See also
    --------
    :class:`~chemprop.v2.models.modules.agg.MeanAggregation`
    :class:`~chemprop.v2.models.modules.agg.SumAggregation`
    :class:`~chemprop.v2.models.modules.agg.NormAggregation`
    """

    def __init__(self, dim: int = 0, *args, **kwargs):
        super().__init__()

        self.dim = dim
        self.hparams = {"dim": dim, "cls": self.__class__}

    @abstractmethod
    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        """Aggregate the graph-level representations of a batch of graphs into their respective
        global representations

        NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
        a zero vector of length `d` in the final output.

        Parameters
        ----------
        H : Tensor
            a tensor of shape ``V x d`` containing the batched node-level representations of ``b``
            graphs
        batch : Tensor
            a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to

        Returns
        -------
        Tensor
            a tensor of shape ``b x d`` containing the graph-level representations
        """


AggregationRegistry = ClassRegistry[Aggregation]()


@AggregationRegistry.register("mean")
class MeanAggregation(Aggregation):
    r"""Average the graph-level representation:

    .. math::
        \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v
    """

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        return scatter_reduce(H, self.dim, batch, H, "mean")


@AggregationRegistry.register("sum")
class SumAggregation(Aggregation):
    r"""Sum the graph-level representation:

    .. math::
        \mathbf h = \sum_{v \in V} \mathbf h_v

    """

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        return scatter_reduce(H, self.dim, batch, H, "sum")


@AggregationRegistry.register("norm")
class NormAggregation(SumAggregation):
    r"""Sum the graph-level representation and divide by a normalization constant:

    .. math::
        \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v
    """

    def __init__(self, dim: int = 0, *args, norm: float = 100, **kwargs):
        super().__init__(dim, **kwargs)

        self.norm = norm
        self.hparams["norm"] = norm

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        return super().forward(H, batch) / self.norm


class AttentiveAggregation(Aggregation):
    def __init__(self, dim: int = 0, *args, output_size: int, **kwargs):
        super().__init__(dim, *args, **kwargs)

        self.W = nn.Linear(output_size, 1)

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        alphas = _scatter_softmax(self.W(H), batch, self.dim)

        return scatter_reduce(alphas * H, self.dim, batch, alphas * H, reduce="sum")
