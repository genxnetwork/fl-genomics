from utils.plink import run_plink
from config.qc_config import sample_qc_config


def sample_qc(bin_file_path: str, output_path: str, bin_file_type='--pfile'):
    """
    Performs sample QC on the whole UKB dataset, writing filtered sample IDs
    to the output path.
    """
    run_plink(
        args_dict={**{bin_file_type: bin_file_path,
                        '--out': output_path},
                     **sample_qc_config},
        args_list=['--write-samples']
    )


class QC(object):
    """ Class that utilises QC to be used for local QC """
    @staticmethod
    def qc(input_prefix: str, qc_config: dict) -> str:
        """ Runs plink command that performs QC """
        output_prefix = input_prefix + '_filtered'
        run_plink(args_list=['--make-pgen'],
                  args_dict={**{'--pfile': input_prefix, # Merging dicts here
                                '--out': output_prefix},
                             **qc_config})
        return output_prefix
