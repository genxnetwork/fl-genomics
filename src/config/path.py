import os.path

ukb_pfile_path = '/gpfs/gpfs0/ukb_data/plink/plink'
ukb_loader_dir = '/gpfs/gpfs0/ukb_data/processed_data/fml'
data_root = '/trinity/home/s.mishra/test'
sample_qc_ids_path = f'{data_root}/passed_sample_qc'
areas_path = '/trinity/home/s.mishra/nuts/UK_division.csv'

TG_DATA_ROOT = '/home/dkolobok/TG/data'
TG_OUT = '/home/dkolobok/TG/out'
TG_BFILE_PATH = os.path.join(TG_DATA_ROOT, 'unphased')
TG_SAMPLE_QC_IDS_PATH = os.path.join(TG_DATA_ROOT, 'passed_sample_qc')
