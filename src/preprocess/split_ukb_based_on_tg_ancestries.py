from utils.plink import run_plink
import sys
from os import path
import logging
from utils.split import Split
from preprocess.train_val_split import CVSplitter
from preprocess.qc import QC
from configs.global_config import data_root
from configs.split_config import tg_split_name, n_tg_nodes, non_iid_split_name
from configs.qc_config import variant_qc_config
from preprocess.splitter import SplitTG

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
logger = logging.getLogger()

tg_split = Split(path.join(data_root, tg_split_name), 'standing_height', n_tg_nodes)
tg_splitter = SplitTG()
logger.info("Saving IDs")
prefix_splits = tg_splitter.split(make_pgen=True)

for local_prefix in prefix_splits:
    logger.info(f'Running local QC for {local_prefix}')
    local_prefix_qc = local_prefix + '_filtered'
    QC.qc(input_prefix=local_prefix, output_prefix=local_prefix_qc, qc_config=variant_qc_config)


splitter = CVSplitter(tg_split)
logger.info("Splitting for CV")
for node_index in range(n_tg_nodes):
    splitter.split_ids(ids_path=None, node_index=node_index, random_state=0)
    
ethnic_split = Split(path.join(data_root, non_iid_split_name), 'standing_height', 1)    
    
for split in [tg_split]:
    logger.info(f"Processing split {split.root_dir}")
    for node_index in range(n_tg_nodes):
        logger.info(f"Node {node_index}")
        for fold_index in range(1):
            for part_name in ['train', 'val', 'test']:
                logger.info(f"Saving genotype")
                # Extract and save genotypes
                run_plink(args_dict={
                '--pfile': split.get_source_pfile_path(node_index=node_index),
                '--keep': split.get_ids_path(node_index=node_index, fold_index=fold_index, part_name=part_name),
                '--out':  split.get_pfile_path(node_index=node_index, fold_index=fold_index, part_name=part_name)
                }, args_list=['--make-pgen', '--threads', '4'])

                logger.info(f"Train PCA")
                # Run PCA on train and save weights for projection
                if part_name == 'train':
                    run_plink(args_list=['--pfile', split.get_pfile_path(node_index=node_index, fold_index=fold_index, part_name=part_name),
                                         '--freq', 'counts',
                                         '--out', split.get_pca_path(node_index=node_index, fold_index=fold_index, part=part_name, ext=''),
                                         '--pca', 'allele-wts', '20', 'approx',
                                         '--threads', '4'],
                              )                
                logger.info(f"PCA Projection")    
                # Use saved PCA weights for all projections
                run_plink(args_list=['--pfile', split.get_pfile_path(node_index=node_index, fold_index=fold_index, part_name=part_name),
                                     '--read-freq', split.get_pca_path(node_index=node_index, fold_index=fold_index, part='train', ext='.acount'),
                                     '--score', split.get_pca_path(node_index=node_index, fold_index=fold_index, part='train', ext='.eigenvec.allele'), '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize', '--score-col-nums', '6-25',
                                     '--out', split.get_pca_path(node_index=node_index, fold_index=fold_index, part=part_name, ext=''),
                                     '--threads', '4']
                          )