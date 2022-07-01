import os
import sys

import pandas as pd

from preprocess.splitter_tg import SplitTG
from utils.loaders import load_plink_pcs
from utils.plink import run_plink
from utils.split import Split
from preprocess.train_val_split import CVSplitter, WBSplitter
from configs.global_config import sample_qc_ids_path, data_root, TG_BFILE_PATH, \
    TG_SAMPLE_QC_IDS_PATH, TG_DATA_ROOT, TG_OUT, SPLIT_DIR, SPLIT_ID_DIR, SPLIT_GENO_DIR
from configs.pca_config import pca_config_tg
from configs.qc_config import sample_qc_config, variant_qc_config
from configs.split_config import non_iid_split_name, uniform_split_config, split_map, uneven_split_shares_list, \
    TG_SUPERPOP_DICT, NUM_FOLDS

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
    
    # # Generate file with sample IDs that pass central QC with plink
    # logger.info(f'Running sample QC and saving valid ids to {TG_SAMPLE_QC_IDS_PATH}')
    # sample_qc(bin_file_path=TG_BFILE_PATH, output_path=TG_SAMPLE_QC_IDS_PATH, bin_file_type='--bfile')
    #
    # logger.info(f'Running global PCA')
    # os.makedirs(os.path.join(TG_DATA_ROOT, 'pca'), exist_ok=True)
    # PCA().run(input_prefix=TG_BFILE_PATH, pca_config=pca_config_tg,
    #           output_path=os.path.join(TG_DATA_ROOT, 'pca', 'global'),
    #           scatter_plot_path=None,
    #           # scatter_plot_path=os.path.join(TG_OUT, 'global_pca.html'),
    #           bin_file_type='--bfile')
    #
    # # Split dataset into IID and non-IID datasets and then QC each local dataset
    # logger.info("Splitting ethnic dataset")
    # prefix_splits = SplitTG().split(make_pgen=True)
    #
    # for local_prefix in prefix_splits:
    #     logger.info(f'Running local QC for {local_prefix}')
    #     local_prefix_qc = QC.qc(input_prefix=os.path.join(SPLIT_GENO_DIR, local_prefix), qc_config=variant_qc_config)
    #
    logger.info("making k-fold split for the TG dataset")
    nodes = list(set(TG_SUPERPOP_DICT.values()))
    superpop_split = Split(SPLIT_DIR, 'ancestry', nodes=nodes)
    splitter = CVSplitter(superpop_split)

    for node in nodes:
        splitter.split_ids(ids_path=os.path.join(SPLIT_GENO_DIR, f'{node}.psam'), node=node, random_state=0)
        
    ancestry_df = SplitTG().get_ethnic_background()
    logger.info(f"Processing split {superpop_split.root_dir}")
    # temporary solution
    all_pca = load_plink_pcs('/media/storage/TG/data/pca/global.eigenvec')
    for node in set(TG_SUPERPOP_DICT.values()):
        logger.info(f"Saving train, val, test genotypes and running PCA for node {node}")
        for fold_index in range(NUM_FOLDS):
            for part_name in ['train', 'val', 'test']:
                ids_path = superpop_split.get_ids_path(node=node, fold_index=fold_index, part_name=part_name)

                # # Extract and save genotypes
                # run_plink(args_dict={
                # '--pfile': superpop_split.get_source_pfile_path(node=node),
                # '--keep': ids,
                # '--out':  superpop_split.get_pfile_path(node=node, fold_index=fold_index, part_name=part_name)
                # }, args_list=['--make-pgen'])

                # write ancestries aka phenotypes
                relevant_ids = ancestry_df['IID'].isin(pd.read_csv(ids_path, sep='\t')['IID'])
                ancestry_df.loc[relevant_ids, ['IID', 'ancestry']].to_csv(superpop_split.get_phenotype_path(node=node, fold_index=fold_index, part=part_name), sep='\t', index=False)

                # temporary solution
                all_pca.reindex(relevant_ids).to_csv(superpop_split.get_pca_path(node=node, fold_index=fold_index, part=part_name))

                # # Run PCA on train and save weights for projection
                # if part_name == 'train':
                #     run_plink(args_list=['--pfile', superpop_split.get_pfile_path(node=node, fold_index=fold_index, part_name=part_name),
                #                          '--freq', 'counts',
                #                          '--out', superpop_split.get_pca_path(node=node, fold_index=fold_index, part=part_name, ext=''),
                #                          '--pca', 'allele-wts', '20'],
                #               )
                # # Use saved PCA weights for all projections
                # run_plink(args_list=['--pfile', superpop_split.get_pfile_path(node=node, fold_index=fold_index, part_name=part_name),
                #                      '--read-freq', superpop_split.get_pca_path(node=node, fold_index=fold_index, part='train', ext='.acount'),
                #                      '--score', superpop_split.get_pca_path(node=node, fold_index=fold_index, part='train', ext='.eigenvec.allele'), '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize', '--score-col-nums', '6-25',
                #                      '--out', superpop_split.get_pca_path(node=node, fold_index=fold_index, part=part_name, ext='')]
                #           )

