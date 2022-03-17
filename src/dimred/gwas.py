from collections import namedtuple
import logging
import shutil
import os
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
    
    print('pythonpath is :' , os.environ['PYTHONPATH'])
    logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

    genotype = snakemake.input['genotype'].replace('.pgen', '')
    phenotype = snakemake.input['phenotype']
    covariates = snakemake.input['covariates'] 
    out_prefix = snakemake.params['out_prefix']
    phenotype_name = snakemake.params['phenotype_name']
    threads = str(snakemake.threads)
    mem_mb = str(snakemake.resources['mem_mb'])
    output_path = snakemake.output['results']

    args = [
            '--pfile', genotype,
            '--pheno', phenotype,
            '--covar', covariates,
            '--glm', 'no-x-sex', 'log10', 'hide-covar',
            '--out', out_prefix,
            '--threads', threads,
            '--memory', mem_mb
    ]
    print('args:')
    print(' '.join(args))
    print()
    run_plink(args)
    
    shutil.move(get_gwas_output_path(out_prefix, phenotype_name, 'real'), output_path)

    logging.info(f'GWAS for genotype {genotype} and phenotype {phenotype_name} finished')