from typing import List
import subprocess

from configs.global_config import PLINK2_BIN, PLINK19_BIN


def run_plink(args_list: List[str] = [], args_dict: dict = None, plink_version: str = '2.0'):
    """Runs plink 2.0 with specified args. Args should NOT contain path to plink2 binary

    Args:
        args_list (List[str]): List of cmd args for plink2.0
        args_dict (dict): Dictionary of cmd args for plink2.0

    Raises:
        RuntimeError: If plink returned a error
    """
    lst = [[k, v] for k, v in args_dict.items()] if args_dict is not None else []
    plink_bin = PLINK19_BIN if plink_version == '1.9' else PLINK2_BIN
    plink = subprocess.run([plink_bin] + args_list + [x for xs in lst for x in xs], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if plink.returncode != 0:
        raise RuntimeError(plink.stderr.decode('utf-8'))


def get_gwas_output_path(output_path: str, phenotype_name: str, phenotype_type: str):
    if phenotype_type == 'binary':
        return f'{output_path}.{phenotype_name}.glm.logistic'
    else:
        return f'{output_path}.{phenotype_name}.glm.linear'
