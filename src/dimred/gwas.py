from collections import namedtuple
import logging
import shutil
import os
import sys
# sys.path.append('/beegfs/home/a.medvedev/uk-biobank/src')
from utils.plink import run_plink, get_gwas_output_path


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        # for isolated testing
        Snakemake = namedtuple('Snakemake', ['input', 'output', 'params', 'resources', 'log'])
        snakemake = Snakemake(
            input={'genotype': 'test.input'},
            output=['test.output'],
            resources={'threads': 1},
            log=['test.log']
        )
    
    print('cwd: ', os.getcwd())
    print('sys.path: ', sys.path)
    print('env:')
    print(os.environ['PYTHONPATH'])
    logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

    genotype = snakemake.input['genotype']
    phenotype = snakemake.input['phenotype'] 
    output = snakemake.output[0]
    threads = snakemake.resources['threads']
    
    args = [
            '--pfile', genotype,
            '--pheno', phenotype,
            '--glm', 'no-x-sex', 'log10', 'allow-no-covars',
            '--out', output,
            '--threads', threads
    ]
#   run_plink(args)
    
#    shutil.move(get_gwas_output_path(output, phenotype, 'real'), 
#        f'{output}.tsv')

#    logging.info(f'GWAS for genotype {genotype} and phenotype {phenotype} finished')