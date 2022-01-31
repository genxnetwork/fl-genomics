from config.path import ukb_loader_dir, valid_ids_path, ukb_bfile_path
# from config.pca_config import pca_config
from config.qc_config import qc_config
from config.split_config import split_config
from preprocess.central_qc import central_qc
# from preprocess.pca import PCA
from preprocess.qc import QC
from preprocess.split import SplitNonIID, SplitIID
from utils.plink import run_plink

import logging
import sys


if __name__ == '__main__':
    # runs the whole pipeline
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger()
    
    # Generate file with sample IDs that pass central QC with UKB fields
    logger.info(f'Saving valid IDs to {valid_ids_path}')
    central_qc(ukb_loader_dir, valid_ids_path)

#     todo: PCA for central dataset
#     PCA.run(input_prefix=prefix_qc, pca_config=pca_config, output_tag='smth')

    # Split dataset into IID and non-IID datasets and then QC + PCA each local dataset
    logger.info("Splitting IID dataset")
    prefix_splits = SplitIID(split_config=split_config).split()
    logger.info("Splitting non-IID dataset")
    prefix_splits += SplitNonIID(split_config=split_config).split()
    for local_prefix in prefix_splits:
        logger.info(f'Running local QC for {local_prefix}')
        local_prefix_qc = QC.qc(input_prefix=local_prefix, qc_config=qc_config)
        # todo
        # PCA.run(input_prefix=local_prefix_qc, pca_config=pca_config, output_tag='smth else')

