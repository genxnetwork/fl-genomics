import os.path
import sys

from configs.global_config import ukb_loader_dir, sample_qc_ids_path, ukb_pfile_path, data_root
from configs.pca_config import pca_config
from configs.qc_config import sample_qc_config, variant_qc_config
from configs.split_config import non_iid_split_name, uniform_split_config, split_map, uneven_split_shares_list
from preprocess.pca import PCA
from preprocess.qc import QC, sample_qc
from preprocess.splitter import SplitNonIID
from utils.plink import run_plink
from utils.split import Split
from preprocess.train_val_split import CVSplitter, WBSplitter

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
    logger.info(f'Running sample QC and saving valid ids to {sample_qc_ids_path}')
    sample_qc(ukb_pfile_path, sample_qc_ids_path)
    
    logger.info(f'Running global PCA')
    PCA().run(input_prefix=ukb_pfile_path, pca_config=pca_config, output_path=os.path.join(data_root, 'pca', 'global'),
              scatter_plot_path=os.path.join(data_root, 'figures', 'global_pca.html'))

    # Split dataset into IID and non-IID datasets and then QC each local dataset
    logger.info("Splitting ethnic dataset")
    prefix_splits = SplitNonIID().split(make_pgen=True)

    for local_prefix in prefix_splits:
        logger.info(f'Running local QC for {local_prefix}')
        local_prefix_qc = QC.qc(input_prefix=local_prefix, qc_config=variant_qc_config)
        
    logger.info("making k-fold split for ethnic dataset")    
    num_ethnic_nodes = max(list(split_map.values()))+1
    ethnic_split = Split(path.join(data_root, non_iid_split_name), 'standing_height', num_ethnic_nodes)
    splitter = CVSplitter(ethnic_split)
    
    for node_index in range(num_ethnic_nodes):
        splitter.split_ids(node_index=node_index, random_state=0)
        
    logger.info("Generating white_british uniform split")
    uniform_split = Split(path.join(data_root, uniform_split_config['uniform_split_name']),
                          'standing_height',
                          uniform_split_config['n_nodes'])
    uniform_splitter = WBSplitter(ethnic_split=ethnic_split,
                                  new_split=uniform_split,
                                  n_nodes=uniform_split_config['n_nodes'],
                                  array_split_arg=uniform_split_config['n_nodes'])
    uniform_splitter.split_ids()
    
    
    logger.info("Generating white_british uneven split")
    uneven_split = Split(path.join(data_root, 'uneven_split'),
                          'standing_height',
                         len(uneven_split_shares_list)+1)
    uneven_splitter = WBSplitter(ethnic_split=ethnic_split,
                                 new_split=uneven_split,
                                 n_nodes=len(uneven_split_shares_list)+1,
                                 array_split_arg=uneven_split_shares_list)
    uneven_splitter.split_ids()
        
    for split in [ethnic_split, uniform_split, uneven_split]:
        logger.info(f"Processing split {split.root_dir}")
        for node_index in range(split.node_count):
            logger.info(f"Saving train, val, test genotypes and running PCA for node {node_index}")
            for fold_index in range(10):
                for part_name in ['train', 'val', 'test']:
                    # Symlinks for test set of uneven and uniform splits
                    if (split in [uniform_split, uneven_split]) and part_name == 'test':
                        # Symlink test white british genotypes
                        if not path.exists(split.get_pfile_path(node_index=node_index, fold_index=fold_index, part_name=part_name)):
                            symlink(ethnic_split.get_pfile_path(node_index=0, fold_index=fold_index, part_name=part_name),
                                    split.get_pfile_path(node_index=node_index, fold_index=fold_index, part_name=part_name))
                        
                        # Symlink test white british PCAs
                        if not path.exists(split.get_pca_path(node_index=node_index, fold_index=fold_index, part=part_name, ext='.sscore')):
                            symlink(ethnic_split.get_pca_path(node_index=0, fold_index=fold_index, part=part_name, ext='.sscore'),
                                    split.get_pca_path(node_index=node_index, fold_index=fold_index, part=part_name, ext='.sscore'))
                            
                    else:
                        # Extract and save genotypes
                        run_plink(args_dict={
                        '--pfile': ethnic_split.get_source_pfile_path(node_index) \
                               if split == ethnic_split else ethnic_split.get_source_pfile_path(0),
                        '--keep': split.get_ids_path(node_index=node_index, fold_index=fold_index, part_name=part_name),
                        '--out':  split.get_pfile_path(node_index=node_index, fold_index=fold_index, part_name=part_name)
                        }, args_list=['--make-pgen'])

                        # Run PCA on train and save weights for projection
                        if part_name == 'train':
                            run_plink(args_list=['--pfile', split.get_pfile_path(node_index=node_index, fold_index=fold_index, part_name=part_name),
                                                 '--freq', 'counts',
                                                 '--out', split.get_pca_path(node_index=node_index, fold_index=fold_index, part=part_name, ext=''),
                                                 '--pca', 'allele-wts', '20', 'approx'],
                                      )                    
                        # Use saved PCA weights for all projections
                        run_plink(args_list=['--pfile', split.get_pfile_path(node_index=node_index, fold_index=fold_index, part_name=part_name),
                                             '--read-freq', split.get_pca_path(node_index=node_index, fold_index=fold_index, part='train', ext='.acount'),
                                             '--score', split.get_pca_path(node_index=node_index, fold_index=fold_index, part='train', ext='.eigenvec.allele'), '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize', '--score-col-nums', '6-25',
                                             '--out', split.get_pca_path(node_index=node_index, fold_index=fold_index, part=part_name, ext='')]
                                  )
    
