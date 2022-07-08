import os
import sys
import pandas as pd

from preprocess.pca import PCA
from preprocess.qc import QC, sample_qc
from preprocess.splitter import SplitNonIID
from preprocess.splitter_tg import SplitTG
from utils.plink import run_plink
from utils.split import Split
from preprocess.train_val_split import CVSplitter, WBSplitter
from config.global_config import sample_qc_ids_path, data_root, TG_BFILE_PATH, \
    TG_SAMPLE_QC_IDS_PATH, TG_DATA_ROOT, TG_OUT, SPLIT_DIR, SPLIT_ID_DIR, SPLIT_GENO_DIR, FOLDS_NUMBER
from config.pca_config import pca_config_tg
from config.qc_config import sample_qc_config, variant_qc_config
from config.split_config import non_iid_split_name, uniform_split_config, split_map, uneven_split_shares_list, \
    TG_SUPERPOP_DICT

import logging
from os import path, symlink

if __name__ == '__main__':
    # runs the whole pipeline
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    logger = logging.getLogger()

    # Generate file with sample IDs that pass central QC with plink
    logger.info(f'Running sample QC and saving valid ids to {TG_SAMPLE_QC_IDS_PATH}')
    sample_qc(bin_file_path=TG_BFILE_PATH, output_path=TG_SAMPLE_QC_IDS_PATH, bin_file_type='--bfile')

    logger.info(f'Running global PCA')
    os.makedirs(os.path.join(TG_DATA_ROOT, 'pca'), exist_ok=True)
    PCA().run(input_prefix=TG_BFILE_PATH, pca_config=pca_config_tg,
              output_path=os.path.join(TG_DATA_ROOT, 'pca', 'global'),
              scatter_plot_path=None,
              # scatter_plot_path=os.path.join(TG_OUT, 'global_pca.html'),
              bin_file_type='--bfile')

    # Split dataset into IID and non-IID datasets and then QC each local dataset
    logger.info("Splitting ethnic dataset")
    prefix_splits = SplitTG().split(make_pgen=True)

    for local_prefix in prefix_splits:
        logger.info(f'Running local QC for {local_prefix}')
        local_prefix_qc = QC.qc(input_prefix=os.path.join(SPLIT_GENO_DIR, local_prefix), qc_config=variant_qc_config)

    logger.info("making k-fold split for the TG dataset")
    nodes = list(set(TG_SUPERPOP_DICT.values()))
    superpop_split = Split(SPLIT_DIR, 'ethnicity', nodes=nodes)
    splitter = CVSplitter(superpop_split)

    for node in nodes:
        splitter.split_ids(ids_path=os.path.join(SPLIT_GENO_DIR, f'{node}.psam'), node=node, random_state=0)

    logger.info(f"Processing split {superpop_split.root_dir}")
    for node in set(TG_SUPERPOP_DICT.values()):
        logger.info(f"Saving train, val, test genotypes and running PCA for node {node}")
        for fold_index in range(FOLDS_NUMBER):
            for part_name in ['train', 'val', 'test']:
                # Extract and save genotypes
                run_plink(args_dict={
                '--pfile': superpop_split.get_source_pfile_path(node=node),
                '--keep': superpop_split.get_ids_path(node=node, fold_index=fold_index, part_name=part_name),
                '--out':  superpop_split.get_pfile_path(node=node, fold_index=fold_index, part_name=part_name)
                }, args_list=['--make-pgen'])

    for fold_index in range(FOLDS_NUMBER):
        # Perform centralized sample ids merge to use it with `--keep` flag in plink
        ids = []

        for node in set(TG_SUPERPOP_DICT.values()):
            ids_filepath = superpop_split.get_ids_path(
                fold_index=fold_index,
                part_name='train',
                node=node
            )

            ids.extend(pd.read_csv(ids_filepath, sep='\t')['IID'].to_list())

        # Store the list of ids inside the super population split file structure
        centralized_ids_filepath = superpop_split.get_ids_path(
            fold_index=fold_index,
            part_name='train',
            node='ALL'  # centralized PCA
        )

        pd.DataFrame({'IID': ids}).to_csv(centralized_ids_filepath, sep='\t', index=False)

    for fold_index in range(FOLDS_NUMBER):
        # Train cetralized PCA
        run_plink(
            args_list=[
                '--pfile', TG_BFILE_PATH,
                '--keep', superpop_split.get_ids_path(fold_index=fold_index, part_name='train', node='ALL'),
                '--freq', 'counts',
                '--out', superpop_split.get_pca_path(node='ALL', fold_index=fold_index, part='train', ext=''),
                '--pca', 'allele-wts', '20'
            ]
        )

        # Project train, test, and val parts
        for node in set(TG_SUPERPOP_DICT.values()):
            for part_name in ['train', 'val', 'test']:
                run_plink(
                    args_list=[
                        '--pfile', superpop_split.get_pfile_path(node=node, fold_index=fold_index, part_name=part_name),
                        '--read-freq', superpop_split.get_pca_path(node='ALL', fold_index=fold_index, part='train', ext='.acount'),
                        '--score', superpop_split.get_pca_path(node='ALL', fold_index=fold_index, part='train', ext='.eigenvec.allele'), '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize', '--score-col-nums', '6-25',
                        '--out', superpop_split.get_pca_path(node=node, fold_index=fold_index, part_name=part_name),
                        '--set-missing-var-ids', '@:#'
                    ]
                )
