from abc import abstractmethod
from ukb_loader import UKBDataLoader
from numpy.random import seed, choice
import pandas as pd
pd.options.mode.chained_assignment = None # Shush
import os

from config.path import data_root, sample_qc_ids_path, ukb_loader_dir, ukb_pfile_path
from config.split_config import non_iid_split_name, split_map, heterogeneous_split_codes, heterogeneous_split_name, n_heterogeneous_nodes, random_seed
from utils.plink import run_plink

class SplitBase(object):
    def __init__(self):
        pass

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
        df = df.loc[~df.ethnic_background.isna()]
        df.ethnic_background = df.ethnic_background.astype('int')
        
        # Include FID/IID fields for plink to play nice with the output files
        df['FID'] = df.index
        df['IID'] = df.index
        # Leave only those samples that passed population QC
        sample_qc_ids = pd.read_table(f'{sample_qc_ids_path}.id', index_col='IID')
        df = df.loc[df.index.intersection(sample_qc_ids.index)]
        
        return df
    
    def make_split_pgen(self, split_id_path: str, prefix: str) -> None:
        run_plink(args_dict={'--pfile': ukb_pfile_path,
                             '--out': prefix,
                             '--keep': split_id_path},
                  args_list=['--make-pgen'])
        
        
class SplitNonIID(SplitBase):
    def split(self, make_pgen=True):
        df = self.get_ethnic_background()
        
        # Drop samples with missing/prefer not to answer ethnic background
        df = df[~df.ethnic_background.isin([-1, -3, 0])]
        # Map ethnic backgrounds to our defined splits
        df['split'] = df.ethnic_background.map(split_map)
        
        
        split_id_dir = os.path.join(data_root, non_iid_split_name, 'split_ids')
        genotype_dir = os.path.join(data_root, non_iid_split_name, 'genotypes')
        genotype_node_dirs = [os.path.join(genotype_dir, f"node_{node_index}")
                              for node_index in range(max(list(split_map.values()))+1)]
        
        for dir_ in [split_id_dir, genotype_dir] + genotype_node_dirs:
            os.makedirs(dir_, exist_ok=True)
        
        prefix_list = []
        for i in range(max(list(split_map.values()))+1):
            split_id_path = os.path.join(split_id_dir, f"{i}.csv")
            prefix = os.path.join(genotype_dir, f"node_{i}")
            prefix_list.append(prefix)
            df.loc[(df.split == i), ['FID', 'IID']].to_csv(split_id_path, index=False, sep='\t')
            
            if make_pgen:
                self.make_split_pgen(split_id_path, prefix)
        
        return prefix_list
        
class SplitHeterogeneous(SplitBase):
    def save_all_ids(self):
        df = self.get_ethnic_background()
        
        # Drop samples with missing/prefer not to answer ethnic background
        df = df[~df.ethnic_background.isin([-1, -3, 0])]
        # Keep ethnicities included in heterogeneous split
        df = df[df.ethnic_background.isin(heterogeneous_split_codes)]
        heterogeneous_ids_path = os.path.join(data_root, heterogeneous_split_name, 'split_ids', '0.csv')
        df.loc[:, ['FID', 'IID']].to_csv(heterogeneous_ids_path, index=False, sep='\t')
        
    def split(self, pca_path):
        df = pd.read_table(pca_path)
        df.columns.values[0] = 'FID'
        model = KMeans(n_heterogeneous_nodes)
        df['cluster'] = model.fit_predict(df.iloc[:, 2:]) + 1
        
        for cluster_index in range(1, n_heterogeneous_nodes+1):
            df.loc[(df.cluster == i), ['FID', 'IID']].to_csv(split_id_path, index=False, sep='\t')
        
        
        