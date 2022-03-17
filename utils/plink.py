from typing import List
import subprocess


def run_plink(args_list=[], args_dict={}):
    """Runs plink 2.0 with specified args. Args should NOT contain path to plink2 binary

    Args:
        args_list (List[str]): List of cmd args for plink2.0
        args_dict: Dict {arg_key: arg_value}

    Raises:
        RuntimeError: If plink returned an error
    """

    if len(args_list) + len(args_dict) == 0:
        raise ValueError("PLINK command has no arguments!")
    plink = subprocess.run(['plink2'] + args_list + [item for key_value in args_dict.items() for item in key_value],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if plink.returncode != 0:
        raise RuntimeError(plink.stderr.decode('utf-8'))
