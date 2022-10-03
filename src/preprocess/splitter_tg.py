import os
import sys

import pandas as pd

from configs.global_config import TG_DATA_ROOT, TG_SAMPLE_QC_IDS_PATH, SPLIT_DIR, SPLIT_ID_DIR, SPLIT_GENO_DIR
from configs.split_config import TG_SUPERPOP_DICT
from preprocess.splitter import SplitBase

import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger()



class SplitTG(SplitBase):
    # def _load_data(self) -> pd.DataFrame:
    # x = pd.read_csv(os.path.join(TG_DATA_ROOT, 'global.eigenvec'), sep='\t').set_index('IID')
    # y = pd.read_csv(os.path.join(TG_DATA_ROOT, 'samples.tsv')).set_index('IID')['pop']
    def __init__(self):
        super().__init__()
        self.nodes = list(set(TG_SUPERPOP_DICT.values()))
        self.nums = list(range(len(self.nodes)))
        self.nodes_num_dict = dict(zip(self.nums, self.nodes))
        self.df = None

    def get_target(self, min_samples_in_pop=30, alpha=1.0) -> pd.DataFrame:
        """
        Loads the ethnic background phenotype for samples that passed initial QC,
        drops rows with missing values and returns a DataFrame with degree of
        dissimilarity equal alpha (alpha = 1 means completely heterogeneous,
        alpha = 0 - fully homogeneous) reformatted to be used
        for downstream analysis with PLINK.
        """

        y = pd.read_csv(os.path.join(TG_DATA_ROOT, 'igsr_samples.tsv'), sep='\t').rename(
            columns={'Sample name': 'IID', 'Population code': 'ancestry', 'Superpopulation code': 'superpop'})

        # Leave only those samples that passed population QC
        sample_qc_ids = pd.read_table(f'{TG_SAMPLE_QC_IDS_PATH}.id')
        y = y.loc[y['IID'].isin(sample_qc_ids['#IID']), :]

        # filter by min number of samples in a pop
        pop_val_counts = y['ancestry'].value_counts()
        y = y[y['ancestry'].isin(pop_val_counts[pop_val_counts >= min_samples_in_pop].index)]

        # Split in homo and hetero parts
        y = y.sample(len(y)).reset_index()
        start = int(len(y) * (1 - alpha))
        heter_part = y.iloc[start:]
        homo_part = y.iloc[:start]

        homo_part['split'] = homo_part['index'] % len(self.nums)
        homo_part = homo_part.replace({"split": self.nodes_num_dict})
        homo_part = homo_part[['IID', 'ancestry', 'split', 'Population name', 'Superpopulation name']]
        homo_counts = homo_part['split'].value_counts()

        heter_part = heter_part.rename(columns={"superpop": "split"})
        heter_part = heter_part[['IID', 'ancestry', 'Population name', 'Superpopulation name', 'split']]
        heter_counts = heter_part['split'].value_counts()

        all_parts = pd.concat([heter_part, homo_part])

        logger.info(f'Heterogeneous values in each node:\n {heter_counts}')
        logger.info(f'Homogeneous values in each node:\n {homo_counts}')
        self.df = all_parts
        return all_parts

    def split(self, input_prefix: str, make_pgen=True, df=None, alpha=1.0):
        if not df:
            df = self.get_target(alpha=alpha)
        # Map ethnic backgrounds to our defined splits
        # df['split'] = df['pop'].map(TG_SUPERPOP_DICT)

        os.makedirs(SPLIT_ID_DIR, exist_ok=True)
        os.makedirs(SPLIT_DIR, exist_ok=True)
        splits_prefixes = df['split'].unique()
        for prefix in splits_prefixes:
            prefix = str(prefix)
            split_id_path = os.path.join(SPLIT_ID_DIR, f"{prefix}.tsv")
            df.loc[df['split'] == prefix, 'IID'].to_csv(split_id_path, index=False, sep='\t', header=False)
            if make_pgen:
                os.makedirs(SPLIT_GENO_DIR, exist_ok=True)
                self.make_split_pgen(split_id_path, out_prefix=os.path.join(SPLIT_GENO_DIR, prefix),
                                     bin_file_type='--pfile', bin_file=input_prefix)

        # create the folder ALL with all data to make runs easier
        split_id_path = os.path.join(SPLIT_ID_DIR, f"ALL.tsv")
        df.loc[:, 'IID'].to_csv(split_id_path, index=False, sep='\t', header=False)
        if make_pgen:
            self.make_split_pgen(split_id_path, out_prefix=os.path.join(SPLIT_GENO_DIR, 'ALL'), bin_file_type='--pfile',
                                 bin_file=input_prefix)
        return list(splits_prefixes)

