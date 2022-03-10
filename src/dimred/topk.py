from collections import namedtuple
import logging
import pandas

from utils.gwas import get_topk_snps
from utils.plink import run_plink


def write_snplist(gwas_path: str, max_snp_count: int):
    """Writes top k SNP IDs by p-value into {gwas_path}.{max_snp_count}.snplist. 

    Args:
        gwas_path (str): Path to plink 2.0 GWAS results with LOG10_P values
        max_snp_count (int): 
    """    
    gwas = pandas.read_table(gwas_path).set_index('ID')
    
    topk_snps = get_topk_snps(gwas, max_snp_count)
    
    snplist_path = gwas_path + f'.{max_snp_count}.snplist'
    with open(snplist_path, 'w') as file:
        file.write('\n'.join(topk_snps.index.tolist()))

    return snplist_path

def write_temp_fid_iid(phenotype_path: str, fid_iid_path: str):
    phenotype = pandas.read_table(phenotype_path)
    phenotype.rename({'FID': '#FID'}, axis='columns', inplace=True)
    phenotype.loc[:, ['#FID', 'IID']].to_csv(fid_iid_path, index=False, sep='\t')

if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        # for isolated testing
        Snakemake = namedtuple('Snakemake', ['input', 'output', 'params', 'threads', 'wildcards', 'resources', 'log'])
        snakemake = Snakemake(
            input={'phenotype': 'test.phenotype', 'gwas': 'test.gwas'},
            output={'pgen': 'test.pfile.out.pgen'},
            params={'in_prefix': 'test.pfile.in', 'out_prefix': 'test.pfile.out'},
            wildcards={'snp_count': '1000'},
            resources={'mem_mb': 1000},
            threads=1,
            log=['test.log']
        )

    logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

    input = snakemake.input[0]

    in_pfile = snakemake.params['in_prefix']
    phenotype_path = snakemake.input['phenotype']
    gwas_path = snakemake.input['gwas']
    max_snp_count = int(snakemake.wildcards['snp_count'])
    threads = str(snakemake.threads)
    mem_mb = str(snakemake.resources['mem_mb'])
    out_pfile = snakemake.params['out_prefix']

    write_temp_fid_iid(phenotype_path, out_pfile + '.samples')

    snplist_path = write_snplist(gwas_path, max_snp_count)
    args = ['--pfile', in_pfile,
            '--extract', snplist_path, '--keep', out_pfile + '.samples', 
            '--threads', threads, '--memory', mem_mb,
            '--make-pgen', '--out', out_pfile]
    run_plink(args)

    print(f'Extracted {max_snp_count} SNPs from {in_pfile} to {out_pfile}')