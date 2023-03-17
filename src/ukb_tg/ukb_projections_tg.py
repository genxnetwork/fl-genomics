### We want to calculate UKB samples projections in TG PC space and apply TG-trained classifier to get
### ancestry predictions for UKB samples
### Note: we want to conduct all actions (PCA, model training) for intersection of UKB and TG variants
### Note2: UKB is in Hg19, TG is in Hg38, in order to intersect variants we filter TG variants by rsids that
### are present in UKB (coordinates differ between assemblies but rsids remain the same and variants in UKB have rsids)
import logging
import os
import sys
import numpy as np
import pandas as pd

from configs.global_config import TG_DATA_ROOT, SPLIT_GENO_DIR

# 0. Preparation
from local.tg_simple_trainer import SimpleTrainer
from preprocess.splitter_tg import SplitTG
from utils.plink import run_plink

TG_EXT_DIR = os.path.join(TG_DATA_ROOT, 'external')
TG_UKB_DIR = os.path.join(TG_EXT_DIR, 'ukb')
TG_UKB_MODELS_DIR = os.path.join(TG_UKB_DIR, 'models')


class UkbProjectionsTg(object):
    def __init__(self, num_pcs=20):
        logger.info(f'Initializing class with {num_pcs} PCs')
        self.num_pcs = num_pcs
        self.tg_pca_prefix = os.path.join(TG_UKB_DIR, 'tg_pca_100')
        self.tg_pca_models_prefix = os.path.join(TG_UKB_MODELS_DIR, 'tg_pca')
        self.tg_pca_prefix_pcs = f'{self.tg_pca_models_prefix}_{num_pcs}pcs'
        self.input_pfile = os.path.join(TG_UKB_DIR, 'tg_filt')

    def main(self):
        # # 1. Take UKB variants file and leave only the rsid column
        # ukb_variants_fn = os.path.join(TG_UKB_DIR, 'ukb_filtered.bim')
        # ukb_rsids_fn = os.path.join(TG_UKB_DIR, 'ukb_rsids.txt')
        # os.system("awk '{print $2}' " + ukb_variants_fn + " > " + ukb_rsids_fn)
        #
        # # 2. Extract intersection variants
        # logger.info(f'Filtering by variants present both in UKB and TG')
        # all_filtered_fn = os.path.join(SPLIT_GENO_DIR, 'ALL_filtered')
        # tg_filt_prefix = os.path.join(TG_UKB_DIR, 'tg_filt')
        # run_plink(
        #     args_list=[
        #         '--pfile', all_filtered_fn,
        #         '--extract', ukb_rsids_fn,
        #         '--make-pgen',
        #         '--out', tg_filt_prefix,
        #     ]
        # )
        #
        # # 3. Conduct PCA on filtered TG data
        # logger.info(f'Running centralised PCA on all TG samples for variants present in UKB and TG')
        # run_plink(
        #     args_list=[
        #         '--pfile', tg_filt_prefix,
        #         '--freq', 'counts',
        #         '--out', self.tg_pca_prefix,
        #         '--pca', 'allele-wts', str(self.num_pcs)
        #     ]
        # )

        # 4. Get TG data projections onto TG's PC space (1 / sqrt(.eigenval) scaling)
        logger.info(f"Getting TG data projections onto TG's PC space (1 / sqrt(.eigenval) scaling)")
        run_plink(
            args_list=[
                '--pfile', self.input_pfile,
                '--read-freq', self.tg_pca_prefix + '.acount',
                '--score', self.tg_pca_prefix + '.eigenvec.allele',
                '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize', '--score-col-nums', f'6-{5 + self.num_pcs}',
                '--out', self.tg_pca_prefix_pcs,
                '--set-missing-var-ids', '@:#'
            ]
        )

        logger.info(f'Done!')

        # 5. Write TG phenotypes
        ancestry_df = SplitTG().get_target()
        relevant_ids = ancestry_df['IID'].isin(pd.read_csv(self.input_pfile + '.psam', sep='\t')['#IID'])
        ancestry_df.loc[relevant_ids, ['IID', 'ancestry']].to_csv(self.tg_pca_prefix + '.tsv', sep='\t', index=False)

        # 6. Train a model
        x = pd.read_csv(self.tg_pca_prefix_pcs + '.sscore', sep='\t').rename(columns={'#IID': 'IID'}).set_index('IID').filter(
            like='_AVG')
        y = pd.read_csv(self.tg_pca_prefix + '.tsv', sep='\t').set_index('IID').reindex(x.index)['ancestry']
        _, y = np.unique(y, return_inverse=True)
        # # CV is commented out because it's needed only to check performance
        logger.info(f'Training CV for {self.num_pcs} PCs...')
        SimpleTrainer(nclass=len(np.unique(y)), nfeat=x.shape[1], epochs=10000, lr=0.05).run_cv(x=x.values, y=y, K=10)
        # logger.info(f'Training full model for {self.num_pcs} PCs...')
        # SimpleTrainer(nclass=len(np.unique(y)), nfeat=x.shape[1], epochs=10000, lr=0.1).train_and_save(x=x.values, y=y, out_fn=f'{self.tg_pca_prefix_pcs}.pkl')
        logger.info(f'Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    logger = logging.getLogger()
    for num_pcs in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        UkbProjectionsTg(num_pcs=num_pcs).main()


