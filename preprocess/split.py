from abc import abstractmethod
from ukb_loader import UKBDataLoader
from numpy.random import seed, choice
import pandas as pd
pd.options.mode.chained_assignment = None # Shush

import sys
sys.path.append('../config')

from config.path import data_root, valid_ids_path, ukb_loader_dir, ukb_pfile_path
from config.split_config import split_map, random_seed
from utils.plink import run_plink

class SplitBase(object):
    def __init__(self, split_config: dict):
        self.split_config = split_config

    @abstractmethod
    def split(self):
        pass
    
    def get_ethnic_background(self) -> pd.DataFrame:
        """
        Loads the ethnic background phenotype for samples that passed initial QC,
        drops rows with missing values and returns a DataFrame formatted to be used
        for downstream analysis with PLINK.
        """
        loader = UKBDataLoader(ukb_loader_dir, 'split', '21000', ['31'])
        df = pd.concat((loader.load_train(), loader.load_val(), loader.load_test()))
        df.columns = ['sex', 'ethnic_background']
        df.loc[df.ethnic_background.isna(), 'ethnic_background'] = 0
        df.ethnic_background = df.ethnic_background.astype('int')
        
        # Include FID/IID fields for plink to play nice with the output files
        df['FID'] = df['IID'] = df.index
        # Leave only those samples that passed population QC
        pop_qc_ids = pd.read_csv(valid_ids_path, index_col='IID', sep='\t')
        df = df.loc[df.index.intersection(pop_qc_ids.index)]
        
        return df
    
    def make_split_pgen(self, split_id_path: str, prefix: str) -> None:
        run_plink(args_dict={'--pfile': ukb_pfile_path,
                             '--out': prefix,
                             '--keep': split_id_path},
                  args_list=['--make-pgen'])

    
class SplitIID(SplitBase):
    def split(self, make_pgen=True):
        df = self.get_ethnic_background()
        # Leave only white british individuals in the IID split
        df = df.loc[df.ethnic_background == 1001]      
        
        seed(random_seed)
        df['split'] = choice(list(range(self.split_config['n_iid_splits'])), size=df.shape[0], replace=True)
        
        prefix_list = []        
        for i in range(self.split_config['n_iid_splits']):
            split_id_path = f"{data_root}/{self.split_config['iid_split_name']}/split_ids/{i}.csv"
            prefix = f"{data_root}/{self.split_config['iid_split_name']}/genotypes/split_{i}"
            prefix_list.append(prefix)
            df.loc[(df.split == i), ['FID', 'IID', 'sex']].to_csv(split_id_path, index=False, sep='\t')
            
            if make_pgen:
                self.make_split_pgen(split_id_path, prefix)
            
        return prefix_list
        
        
class SplitNonIID(SplitBase):
    def split(self, make_pgen=True):
        df = self.get_ethnic_background()
        
        # Drop samples with missing/prefer not to answer ethnic background
        df = df[~df.ethnic_background.isin([-1, -3, 0])]
        # Map ethnic backgrounds to our defined splits
        df['split_code'] = df.ethnic_background.map(split_map)
        num_test_split = max(list(split_map.values())) + 1
        
        seed(random_seed)
        holdout_idx = choice(df.index, size=int(df.shape[0]*self.split_config['non_iid_holdout_ratio']), replace=False)
        
        df['split'] = df['split_code'].copy()
        df.loc[holdout_idx, 'split'] = num_test_split
        
        prefix_list = []
        for i in range(num_test_split+1):
            split_id_path = f"{data_root}/{self.split_config['non_iid_split_name']}/split_ids/{i}.csv"
            prefix = f"{data_root}/{self.split_config['non_iid_split_name']}/genotypes/split_{i}"
            prefix_list.append(prefix)
            df.loc[(df.split == i), ['FID', 'IID', 'sex']].to_csv(split_id_path, index=False, sep='\t')
            
            if make_pgen:
                self.make_split_pgen(split_id_path, prefix)
        
        return prefix_list
        
