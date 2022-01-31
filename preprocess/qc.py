import sys
sys.path.append('../utils')

from utils.plink import run_plink

class QC(object):
    """ Class that utilises QC to be used for local QC """
    @staticmethod
    def qc(input_prefix: str, qc_config: dict) -> str:
        """ Runs plink command that performs QC """
        run_plink(args_dict=qc_config)
                
        output_prefix = input_prefix + '_filtered'
        return output_prefix
