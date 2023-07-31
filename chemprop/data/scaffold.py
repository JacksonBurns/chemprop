import logging
from typing import List, Set, Tuple
import warnings
import numpy as np

from .data import MoleculeDataset


def log_scaffold_stats(data: MoleculeDataset,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular scaffolds.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_scaffolds: The number of scaffolds about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :param logger: A logger for recording output.
    :return: A list of tuples where each tuple contains a list of average target values
             across the first :code:`num_labels` labels and a list of the number of non-zero values for
             the first :code:`num_scaffolds` scaffolds, sorted in decreasing order of scaffold frequency.
    """
    if logger is not None:
        logger.debug('Label averages per scaffold, in decreasing order of scaffold frequency,'
                     f'capped at {num_scaffolds} scaffolds and {num_labels} labels:')

    stats = []
    index_sets = sorted(index_sets, key=lambda idx_set: len(idx_set), reverse=True)
    for scaffold_num, index_set in enumerate(index_sets[:num_scaffolds]):
        data_set = [data[i] for i in index_set]
        targets = np.array([d.targets for d in data_set], dtype=float)

        with warnings.catch_warnings():  # Likely warning of empty slice of target has no values besides NaN
            warnings.simplefilter('ignore', category=RuntimeWarning)
            target_avgs = np.nanmean(targets, axis=0)[:num_labels]

        counts = np.count_nonzero(~np.isnan(targets), axis=0)[:num_labels]
        stats.append((target_avgs, counts))

        if logger is not None:
            logger.debug(f'Scaffold {scaffold_num}')
            for task_num, (target_avg, count) in enumerate(zip(target_avgs, counts)):
                logger.debug(f'Task {task_num}: count = {count:,} | target average = {target_avg:.6f}')
            logger.debug('\n')

    return stats
