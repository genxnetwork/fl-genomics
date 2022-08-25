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

from configs.global_config import TG_DATA_CHIP_ROOT, TG_BFILE_PATH, SPLIT_GENO_DIR, TG_DATA_DIR

# 0. Preparation
from local.tg_simple_trainer import SimpleTrainer
from preprocess.splitter_tg import SplitTGHeter
from utils.plink import run_plink

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    logger = logging.getLogger()

    TG_EXT_DIR = os.path.join(TG_DATA_CHIP_ROOT, 'external')
    TG_UKB_DIR = os.path.join(TG_EXT_DIR, 'ukb')

    # 1. Take UKB variants file and leave only the rsid column
    ukb_variants_fn = os.path.join(TG_UKB_DIR, 'ukb_filtered.bim')
    ukb_rsids_fn = os.path.join(TG_UKB_DIR, 'ukb_rsids.txt')
    os.system("awk '{print $2}' " + ukb_variants_fn + " > " + ukb_rsids_fn)

    # 2. Extract intersection variants
    logger.info(f'Filtering by samples present both in UKB and TG')
    all_filtered_fn = os.path.join(SPLIT_GENO_DIR, 'ALL_filtered')
    tg_filt_prefix = os.path.join(TG_UKB_DIR, 'tg_filt')
    run_plink(
        args_list=[
            '--pfile', all_filtered_fn,
            '--extract', ukb_rsids_fn,
            '--make-pgen',
            '--out', tg_filt_prefix,
        ]
    )

    # 3. Conduct PCA on filtered TG data
    logger.info(f'Running centralised PCA on all TG samples for variants present in UKB and TG')
    tg_pca_prefix = os.path.join(TG_UKB_DIR, 'tg_pca')
    run_plink(
        args_list=[
            '--pfile', tg_filt_prefix,
            '--freq', 'counts',
            '--out', tg_pca_prefix,
            '--pca', 'allele-wts', '20'
        ]
    )

    # 4. Get TG data projections onto TG's PC space (1 / sqrt(.eigenval) scaling)
    logger.info(f"Getting TG data projections onto TG's PC space (1 / sqrt(.eigenval) scaling)")
    run_plink(
        args_list=[
            '--pfile', all_filtered_fn,
            '--read-freq', tg_pca_prefix + '.acount',
            '--score', tg_pca_prefix + '.eigenvec.allele',
            '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize', '--score-col-nums', '6-25',
            '--out', tg_pca_prefix,
            '--set-missing-var-ids', '@:#'
        ]
    )

    logger.info(f'Done!')

    # 5. Write TG phenotypes
    ancestry_df = SplitTGHeter().get_target()
    relevant_ids = ancestry_df['IID'].isin(pd.read_csv(all_filtered_fn + '.psam', sep='\t')['#IID'])
    ancestry_df.loc[relevant_ids, ['IID', 'ancestry']].to_csv(tg_pca_prefix + '.tsv', sep='\t', index=False)

# 6. Train a model
tg_pca_prefix = os.path.join(TG_UKB_DIR, 'tg_pca')
x = pd.read_csv(tg_pca_prefix + '.sscore', sep='\t').rename(columns={'#IID': 'IID'}).set_index('IID').filter(like='_AVG')
y = pd.read_csv(tg_pca_prefix + '.tsv', sep='\t').set_index('IID').reindex(x.index)['ancestry']
_, y = np.unique(y, return_inverse=True)
# # CV is commented out because it's needed only to check performance
SimpleTrainer(nclass=len(np.unique(y)), nfeat=x.shape[1], epochs=10000, lr=0.1).run_cv(x=x.values, y=y, K=10)
# SimpleTrainer(nclass=len(np.unique(y)), nfeat=x.shape[1], epochs=10000, lr=0.1).train_and_save(x=x.values, y=y, out_fn=tg_pca_prefix + '.pkl')
logger.info(f'Done!')

