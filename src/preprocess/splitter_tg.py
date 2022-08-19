import os

import pandas as pd

from configs.global_config import TG_DATA_ROOT, TG_SAMPLE_QC_IDS_PATH, TG_BFILE_PATH, SPLIT_DIR, SPLIT_ID_DIR, \
    SPLIT_GENO_DIR
from configs.split_config import TG_SUPERPOP_DICT
from preprocess.splitter import SplitBase


class SplitTG(SplitBase):
    # def _load_data(self) -> pd.DataFrame:
        # x = pd.read_csv(os.path.join(TG_DATA_ROOT, 'global.eigenvec'), sep='\t').set_index('IID')
        # y = pd.read_csv(os.path.join(TG_DATA_ROOT, 'samples.tsv')).set_index('IID')['pop']

    @staticmethod
    def get_ethnic_background(min_samples_in_pop = 30) -> pd.DataFrame:
        """
        Loads the ethnic background phenotype for samples that passed initial QC,
        drops rows with missing values and returns a DataFrame formatted to be used
        for downstream analysis with PLINK.
        """
        y = pd.read_csv(os.path.join(TG_DATA_ROOT, 'igsr_samples.tsv'), sep='\t').rename(columns={'Sample name': 'IID', 'Population code': 'ancestry', 'Superpopulation code': 'split'})
        # Leave only those samples that passed population QC
        sample_qc_ids = pd.read_table(f'{TG_SAMPLE_QC_IDS_PATH}.id')
        y = y.loc[y['IID'].isin(sample_qc_ids['#IID']), :]
        # filter by min number of samples in a pop
        pop_val_counts = y['ancestry'].value_counts()
        y = y[y['ancestry'].isin(pop_val_counts[pop_val_counts >= min_samples_in_pop].index)]
        return y[['IID', 'ancestry', 'split', 'Population name', 'Superpopulation name']]

    def split(self, input_prefix: str, make_pgen=True):
        df = self.get_ethnic_background()

        # Map ethnic backgrounds to our defined splits
        # df['split'] = df['pop'].map(TG_SUPERPOP_DICT)

        os.makedirs(SPLIT_ID_DIR, exist_ok=True)
        os.makedirs(SPLIT_DIR, exist_ok=True)

        for prefix in set(TG_SUPERPOP_DICT.values()):
            split_id_path = os.path.join(SPLIT_ID_DIR, f"{prefix}.tsv")
            df.loc[df['split'] == prefix, 'IID'].to_csv(split_id_path, index=False, sep='\t', header=False)
            if make_pgen:
                os.makedirs(SPLIT_GENO_DIR, exist_ok=True)
                self.make_split_pgen(split_id_path, out_prefix=os.path.join(SPLIT_GENO_DIR, prefix), bin_file_type='--pfile', bin_file=input_prefix)

        # create the folder ALL with all data to make runs easier
        split_id_path = os.path.join(SPLIT_ID_DIR, f"ALL.tsv")
        df.loc[:, 'IID'].to_csv(split_id_path, index=False, sep='\t', header=False)
        if make_pgen:
            self.make_split_pgen(split_id_path, out_prefix=os.path.join(SPLIT_GENO_DIR, 'ALL'), bin_file_type='--pfile',
                                 bin_file=input_prefix)
        return list(set(TG_SUPERPOP_DICT.values())) + ['ALL']
