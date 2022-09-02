import os

import pandas as pd

from utils.plink import run_plink


def run_gwas_subsample(psam_path, n_samples, n_times, out_dir, plink_args_list):
    """
     Randomly selects a subset of samples and perform GWAS on them n times
    :param psam_path: .psam file to take a full list of samples from
    :param n_samples: Number of samples to subsample
    :param n_times: Number of times to perform subsampling and GWAS
    :param out_dir: Folder to write subsample files and to GWAS results
    :param plink_args_list: other arguments for running plink
    :return:
    """
    ser = pd.read_csv(psam_path, sep='\t')['#IID']
    os.makedirs(out_dir, exist_ok=True)
    for i in n_times:
        out_fn = os.path.join(out_dir, f'{n_samples}_{i}')
        ser.sample(n_samples).to_csv(out_fn + '.txt', sep='\t', index=False, header=False)
        run_plink(args_list=plink_args_list, args_dict={'--keep': out_fn + '.txt', '--out': out_fn})
