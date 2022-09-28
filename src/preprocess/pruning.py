import os
import subprocess

from utils.plink import run_plink


class PlinkPruningRunner(object):
    """
    Runs plink pruing using a list of node files to get a single unified list of variant ids.
    """

    def __init__(self, source_directory, nodes, result_filepath, node_filename_template='%s_filtered'):
        self.result_filepath = result_filepath
        self.node_files = []
        for node in nodes:
            filename = node_filename_template % node
            self.node_files.append(os.path.join(source_directory, filename))

    def run(self, window_size: int, step: int, threshold: float):
        """
        Creates two lists of variants which linkage disequilibrium (LD) is above
        the threshold in *.prune.out and below the threshold in *.prune.in.

        Variant ids in *.prune.in files are considered to pass pruning procedure
        and should be used in a downstream analysis.

        These files are joined into a single file: self.result_filepath.
        """

        if not 0 <= threshold <= 1:
            raise ValueError(f'{threshold} is an invalid threshold value! It should be set between 0 and 1.')

        for node_file in self.node_files:
            run_plink(
                args_list=[
                    '--pfile', node_file,
                    '--out', node_file,
                    '--indep-pairwise', f'{window_size}', f'{step}', f'{threshold}'
                ]
            )

        files = ' '.join(file + '.prune.in' for file in self.node_files)
        with open(self.result_filepath, 'w') as result_file:
            subprocess.run([f'cat {files} | sort | uniq'], shell=True, stdout=result_file)
