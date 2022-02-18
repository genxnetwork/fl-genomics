from config.path import ukb_loader_dir, sample_qc_ids_path, ukb_pfile_path, data_root
from config.pca_config import pca_config
from config.qc_config import sample_qc_config
from config.split_config import non_iid_split_config, split_map
from preprocess.pca import PCA
from preprocess.qc import QC, sample_qc
from preprocess.split import SplitNonIID, SplitIID
from utils.plink import run_plink
import sys
sys.path.append('dimred/src')
from utils.split import Split
from gwas.train_val_split import CVSplitter

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
    
    # Generate file with sample IDs that pass central QC with UKB fields
    logger.info(f'Running sample QC and saving valid ids to {sample_qc_ids_path}')
    sample_qc(ukb_pfile_path, sample_qc_ids_path)
    
    logger.info(f'Running global PCA')
    PCA().run(input_prefix=ukb_pfile_path, pca_config=pca_config, output_tag='global')

    # Split dataset into IID and non-IID datasets and then QC + PCA each local dataset
    logger.info("Splitting ethnic dataset")
    prefix_splits = SplitNonIID(split_config=non_iid_split_config).split(make_pgen=False)

    logger.info("making k-fold split for ethnic dataset")    
    num_ethnic_nodes = max(list(split_map.values()))+1
    split = Split(join(data_root, non_iid_split_config['non_iid_split_name']), 'standing_height', num_ethnic_nodes)
    splitter = CVSplitter(split)
    for node_index in range(num_ethnic_nodes):
        splitter.split_ids(node_index, random_state=0)
        
    for local_prefix in prefix_splits:
        logger.info(f'Running local QC for {local_prefix}')
        local_prefix_qc = QC.qc(input_prefix=local_prefix, qc_config=qc_config)
        logger.info(f'Running local PCA for {local_prefix_qc}')
        tag = '_'.join([local_prefix_qc.split('/')[i] for i in [-3, -1]]) # Tag by split name and number
        PCA().run(input_prefix=local_prefix_qc, pca_config=pca_config, output_tag=tag)
