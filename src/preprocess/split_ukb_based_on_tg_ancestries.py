from utils.plink import run_plink
import sys
from os import path
import logging
from utils.split import Split
from preprocess.train_val_split import CVSplitter
from preprocess.qc import QC
from configs.global_config import data_root
from configs.split_config import tg_split_name, n_tg_nodes, n_subsample_nodes, non_iid_split_name
from configs.qc_config import variant_qc_config
from preprocess.splitter import SplitTG

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
logger = logging.getLogger()

tg_split = Split(path.join(data_root, tg_split_name), 'standing_height', n_tg_nodes+n_subsample_nodes)
tg_splitter = SplitTG()
logger.info("Saving IDs")
prefix_splits = tg_splitter.split(make_pgen=True)

for local_prefix in prefix_splits:
    logger.info(f'Running local QC for {local_prefix}')
    local_prefix_qc = local_prefix + '_filtered'
    QC.qc(input_prefix=local_prefix, output_prefix=local_prefix_qc, qc_config=variant_qc_config)


splitter = CVSplitter(tg_split)
logger.info("Splitting for CV")
for node_index in range(n_tg_nodes+n_subsample_nodes):
    splitter.split_ids(ids_path=None, node_index=node_index, random_state=0)
