from config.path import ukb_loader_dir, sample_qc_ids_path, ukb_pfile_path, data_root
from config.pca_config import pca_config
from config.qc_config import sample_qc_config, variant_qc_config
from config.split_config import non_iid_split_name, uniform_split_config, split_map, uneven_split_shares_list
from preprocess.pca import PCA
from preprocess.qc import QC, sample_qc
from preprocess.split import SplitNonIID
from utils.plink import run_plink
import sys
sys.path.append('dimred/src')
from utils.split import Split
from gwas.train_val_split import CVSplitter, WBSplitter

import logging
from os.path import join


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
    PCA().run(input_prefix=ukb_pfile_path, pca_config=pca_config, output_tag='global')

    # Split dataset into IID and non-IID datasets and then QC each local dataset
    logger.info("Splitting ethnic dataset")
    prefix_splits = SplitNonIID().split(make_pgen=True)

    for local_prefix in prefix_splits:
        logger.info(f'Running local QC for {local_prefix}')
        local_prefix_qc = QC.qc(input_prefix=local_prefix, qc_config=variant_qc_config)
        
    logger.info("making k-fold split for ethnic dataset")    
    num_ethnic_nodes = max(list(split_map.values()))+1
    ethnic_split = Split(join(data_root, non_iid_split_name), 'standing_height', num_ethnic_nodes)
    splitter = CVSplitter(ethnic_split)
    
    for node_index in range(num_ethnic_nodes):
        splitter.split_ids(node_index, random_state=0)
        
    logger.info("Generating white_british uniform split")
    uniform_split = Split(join(data_root, uniform_split_config['uniform_split_name']),
                          'standing_height',
                          uniform_split_config['n_nodes'])
    uniform_splitter = WBSplitter(ethnic_split=ethnic_split,
                                  new_split=uniform_split,
                                  n_nodes=uniform_split_config['n_nodes'],
                                  array_split_arg=uniform_split_config['n_nodes'])
    uniform_splitter.split_ids()
    
    
    logger.info("Generating white_british uneven split")
    uneven_split = Split(join(data_root, 'uneven_split'),
                          'standing_height',
                         len(uneven_split_shares_list)+1)
    uneven_splitter = WBSplitter(ethnic_split=ethnic_split,
                                 new_split=uneven_split,
                                 n_nodes=len(uneven_split_shares_list)+1,
                                 array_split_arg=uneven_split_shares_list)
    uneven_splitter.split_ids()
        
    for split, n_nodes in zip([ethnic_split,
                              uniform_split,
                                uneven_split],
                              [5, 5, 8]):
        logger.info(f"split {split.root_dir}")
        for node_index in range(n_nodes):
            logger.info(f"Saving train, val, test genotypes and runnin PCA for node {node_index}")
            for fold_index in range(10):
                for part_name in ['train', 'val', 'test']:
                    run_plink(args_dict={
                    '--pfile': ethnic_split.get_source_pfile_path(node_index) \
                           if split == ethnic_split else ethnic_split.get_source_pfile_path(0),
                    '--keep': split.get_ids_path(node_index, fold_index, part_name),
                    '--out':  split.get_pfile_path(node_index, fold_index, part_name)
                    }, args_list=['--make-pgen'])
                
                    if part_name == 'train':
                        run_plink(args_list=['--pfile', split.get_pfile_path(node_index, fold_index, part_name),
                                             '--out', split.get_pca_path(node_index, fold_index, part_name, ext='')],
                                  args_dict=pca_config)
    
