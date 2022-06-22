import os

import pandas as pd

from config.global_config import TG_DATA_ROOT, TG_SAMPLE_QC_IDS_PATH, TG_BFILE_PATH
from config.split_config import TG_SUPERPOP_DICT
from preprocess.splitter import SplitBase


class SplitTG(SplitBase):
    # def _load_data(self) -> pd.DataFrame:
        # x = pd.read_csv(os.path.join(TG_DATA_ROOT, 'global.eigenvec'), sep='\t').set_index('IID')
        # y = pd.read_csv(os.path.join(TG_DATA_ROOT, 'samples.tsv')).set_index('IID')['pop']

    def get_ethnic_background(self) -> pd.DataFrame:
        """
        Loads the ethnic background phenotype for samples that passed initial QC,
        drops rows with missing values and returns a DataFrame formatted to be used
        for downstream analysis with PLINK.
        """
        y = pd.read_csv(os.path.join(TG_DATA_ROOT, 'samples.tsv'), sep='\t')
        y['IID'] = y['sample']
        y['FID'] = y['sample']
        # Leave only those samples that passed population QC
        sample_qc_ids = pd.read_table(f'{TG_SAMPLE_QC_IDS_PATH}.id')
        y = y.loc[y['IID'].isin(sample_qc_ids['IID']), :]
        return y[['FID', 'IID', 'pop']]

    def split(self, make_pgen=True):
        df = self.get_ethnic_background()

        # Map ethnic backgrounds to our defined splits
        df['split'] = df['pop'].map(TG_SUPERPOP_DICT)

        split_id_dir = os.path.join(TG_DATA_ROOT, 'superpop_split', 'split_ids')
        os.makedirs(split_id_dir, exist_ok=True)
        genotype_dir = os.path.join(TG_DATA_ROOT, 'superpop_split', 'genotypes')
        os.makedirs(genotype_dir, exist_ok=True)

        for prefix in set(TG_SUPERPOP_DICT.values()):
            split_id_path = os.path.join(split_id_dir, f"{prefix}.tsv")
            df.loc[df['split'] == prefix, ['FID', 'IID']].to_csv(split_id_path, index=False, sep='\t')
            if make_pgen:
                self.make_split_pgen(split_id_path, prefix=prefix, bin_file_type='--bfile', bin_file=TG_BFILE_PATH)

        return list(set(TG_SUPERPOP_DICT.values()))
