from os import path
from os import system

from utils.plink import run_plink

from configs.split_config import TG_SUPERPOP_DICT
from configs.global_config import SPLIT_GENO_DIR

class Pruning(object):

    @staticmethod
    def prune(NODE: str, window_size: int, step: int, threshold: float):
        """
        Creates lists of variants which LD is above
        threshold in NODE.preprune.out and below threshold in NODE.preprune.in
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f'{threshold} is an invalid threshold value! It should be set between 0 and 1!')

        run_plink(
            args_list=[
                '--pfile', path.join(SPLIT_GENO_DIR, f'{NODE}'),
                '--out', path.join(SPLIT_GENO_DIR, f'{NODE}.preprune'),
                '--rm-dup',
                '--set-missing-var-ids', '@:#',
                '--make-pgen'
            ]
        )

        run_plink(
            args_list=[
                '--pfile', path.join(SPLIT_GENO_DIR, f'{NODE}.preprune'),
                '--out', path.join(SPLIT_GENO_DIR, f'{NODE}'),
                '--indep-pairwise', f'{window_size}', f'{step}', f'{threshold}'
            ]
        )

    @staticmethod
    def merge(node: str = 'ALL'):
        """
        Merges all nodes in TG_SUPERPOP_DICT NODE.preprune.in lists into one.
        """
        nodes = set(TG_SUPERPOP_DICT.values())
        nodes = (path.join(SPLIT_GENO_DIR, node + '_filtered.prune.in') for node in nodes)
        system(f"cat {' '.join(nodes)} | sort | uniq > {path.join(SPLIT_GENO_DIR, f'{node}.prune.in')}")


    @staticmethod
    def remove_variants(NODE: str, SNPS: str):
        """
        Removes variants from NODE pfiles that are in list SNPS.prune.in
        """
        run_plink(
            args_list=[
                '--pfile', path.join(SPLIT_GENO_DIR, NODE),
                '--out', path.join(SPLIT_GENO_DIR, f'{NODE}.pruned'),
                '--extract', path.join(SPLIT_GENO_DIR, f'{SNPS}.prune.in'),
                '--make-pgen'
            ]
        )




