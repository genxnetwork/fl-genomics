from utils.plink import run_plink
from configs.qc_config import sample_qc_config


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
    def qc(input_prefix: str, output_prefix: str, qc_config: dict) -> str:
        """ Runs plink command that performs QC """
        run_plink(args_list=['--make-pgen', '--hwe', '0.000001', 'midp', 'keep-fewhet'],
                  args_dict={**{'--pfile': input_prefix, # Merging dicts here
                                '--out': output_prefix,
                                '--set-missing-var-ids': '@:#'},
                             **qc_config})
