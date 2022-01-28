from config.path import ukb_raw_fn
from config.pca_config import pca_config
from config.qc_config import qc_config
from config.split_config import split_config
from preprocess.load import ukb_into_plink_binary
from preprocess.pca import PCA
from preprocess.qc import QC
from preprocess.split import SplitNonIID, SplitIID

if __name__ == '__main__':
    # runs the whole pipeline

    # convert data into PLINK format
    prefix_raw = ukb_into_plink_binary(ukb_raw_fn)

    # QC and PCA for central dataset
    prefix_qc = QC.qc(input_prefix=prefix_raw, qc_config=qc_config)
    PCA.run(input_prefix=prefix_qc, pca_config=pca_config, output_tag='smth')

    # Split dataset into IID datasets and then QC + PCA each local dataset
    prefix_splits = SplitIID(split_config=split_config).split()
    for local_prefix in prefix_splits:
        local_prefix_qc = QC.qc(input_prefix=local_prefix, qc_config=qc_config)
        PCA.run(input_prefix=local_prefix_qc, pca_config=pca_config, output_tag='smth else')

    # Split dataset into non-IID datasets and then QC + PCA each local dataset
    prefix_splits = SplitNonIID(split_config=split_config).split()
    for local_prefix in prefix_splits:
        local_prefix_qc = QC.qc(input_prefix=local_prefix, qc_config=qc_config)
        PCA.run(input_prefix=local_prefix_qc, pca_config=pca_config, output_tag='smth else')
