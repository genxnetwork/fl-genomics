from typing import List
import subprocess


def run_plink(args: List[str]):
    """Runs plink 2.0 with specified args. Args should NOT contain path to plink2 binary

    Args:
        args (List[str]): List of cmd args for plink2.0

    Raises:
        RuntimeError: If plink returned a error
    """    
    plink = subprocess.run(['plink2'] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if plink.returncode != 0:
        raise RuntimeError(plink.stderr.decode('utf-8'))


def get_gwas_output_path(output_path: str, phenotype_name: str, phenotype_type: str):
    if phenotype_type == 'binary':
        return f'{output_path}.{phenotype_name}.glm.logistic.firth'
    else:
        return f'{output_path}.{phenotype_name}.glm.linear'