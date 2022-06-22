from utils.plink import run_plink
import sys
from os import path
import logging
from utils.split import Split
from preprocess.train_val_split import CVSplitter
from config.global_config import data_root
from config.split_config import heterogeneous_split_name, n_heterogeneous_nodes, non_iid_split_name
from preprocess.splitter import SplitHeterogeneous

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
logger = logging.getLogger()

heterogeneous_split = Split(path.join(data_root, heterogeneous_split_name), 'standing_height', n_heterogeneous_nodes)
heterogeneous_splitter = SplitHeterogeneous()
logger.info("Saving IDs")
heterogeneous_splitter.split()
splitter = CVSplitter(heterogeneous_split)
for node_index in range(n_heterogeneous_nodes):
    splitter.split_ids(node_index, random_state=0)
    
ethnic_split = Split(path.join(data_root, non_iid_split_name), 'standing_height', 1)    
    
for split in [heterogeneous_split]:
    logger.info(f"Processing split {split.root_dir}")
    for node_index in range(n_heterogenous_nodes):
        logger.info(f"Node {node_index}")
        for fold_index in range(1):
            for part_name in ['train', 'val', 'test']:
                logger.info(f"Saving genotype")
                # Extract and save genotypes
                run_plink(args_dict={
                '--pfile': ethnic_split.get_source_pfile_path(0),
                '--keep': split.get_ids_path(node_index, fold_index, part_name),
                '--out':  split.get_pfile_path(node_index, fold_index, part_name)
                }, args_list=['--make-pgen', '--threads', '4'])

                logger.info(f"Train PCA")
                # Run PCA on train and save weights for projection
                if part_name == 'train':
                    run_plink(args_list=['--pfile', split.get_pfile_path(node_index, fold_index, part_name),
                                         '--freq', 'counts',
                                         '--out', split.get_pca_path(node_index, fold_index, part_name, ext=''),
                                         '--pca', 'allele-wts', '20', 'approx',
                                         '--threads', '4'],
                              )                
                logger.info(f"PCA Projection")    
                # Use saved PCA weights for all projections
                run_plink(args_list=['--pfile', split.get_pfile_path(node_index, fold_index, part_name),
                                     '--read-freq', split.get_pca_path(node_index, fold_index, 'train', ext='.acount'),
                                     '--score', split.get_pca_path(node_index, fold_index, 'train', ext='.eigenvec.allele'), '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize', '--score-col-nums', '6-25',
                                     '--out', split.get_pca_path(node_index, fold_index, part_name, ext=''),
                                     '--threads', '4']
                          )