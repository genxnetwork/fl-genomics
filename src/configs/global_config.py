""" Global configs contains paths and other general settings. This file is added to git.
    Imports local_config.py that is not added to git and
    contains local settings to overwrite those of the global configs
"""
import os
if os.path.exists(os.path.join(os.path.dirname(__file__), 'local_config.py')):
    from configs.local_config import *

###############################################################################
# PATHS
###############################################################################
ukb_pfile_path = globals().get('ukb_pfile_path', '/gpfs/gpfs0/ukb_data/plink/plink')
ukb_loader_dir = globals().get('ukb_loader_dir', '/gpfs/gpfs0/ukb_data/processed_data/fml')
data_root = globals().get('data_root', '/trinity/home/s.mishra/test')
sample_qc_ids_path = globals().get('sample_qc_ids_path', f'{data_root}/passed_sample_qc')
areas_path = globals().get('areas_path', '/trinity/home/s.mishra/nuts/UK_division.csv')

PLINK2_BIN = globals().get('PLINK2_BIN', 'plink2')

TG_DATA_ROOT = globals().get('TG_DATA_ROOT', '/mount/storage/TG/data/chip')
TG_OUT = globals().get('TG_OUT', '/home/dkolobok/TG/out')
TG_BFILE_PATH = globals().get('TG_BFILE_PATH', os.path.join(TG_DATA_ROOT, 'tg'))
TG_SAMPLE_QC_IDS_PATH = globals().get('TG_SAMPLE_QC_IDS_PATH', os.path.join(TG_DATA_ROOT, 'passed_sample_qc'))
SPLIT_DIR = os.path.join(TG_DATA_ROOT, 'superpop_split')
SPLIT_GENO_DIR = os.path.join(TG_DATA_ROOT, 'superpop_split', 'genotypes')
SPLIT_ID_DIR = os.path.join(TG_DATA_ROOT, 'superpop_split', 'split_ids')
