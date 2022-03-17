from collections import namedtuple
from typing import List, Set
import logging
import pandas

from utils.gwas import read_all_gwas, get_topk_snps


def _get_selection_snp_ids(gwas_data: List[pandas.DataFrame], max_snp_count: int) -> Set:
    """Gets SNPs presented in all node datasets, intersects them, 
    and generates a union from top {max_snp_count} most significant SNPs from each node

    Args:
        gwas_data (List[pandas.DataFrame]): GWAS results for each node
        max_snp_count (int): Number of most significant SNPs to be taken from each node

    Returns:
        Set: set of SNP IDs. It will have more than {max_snp_count} SNPs in it.
    """    
    intersection_ids = set(gwas_data[0].index)
    for gwas in gwas_data[1:]:
        intersection_ids &= set(gwas.index)

    topk_data = [get_topk_snps(gwas, max_snp_count) for gwas in gwas_data]
        
    union = pandas.concat(topk_data, join='inner', axis='index')
    union_ids = set(union.index)

    # union.sort_values(by=['#CHROM', 'POS'], inplace=True, ascending=True)
    selection = intersection_ids & union_ids
    
    print(f'intersection of SNPs before GWAS for all datasets has {len(intersection_ids)} SNPs. max_snp_count = {max_snp_count}')
    print(f'topk strategy: union of SNPs contains {len(union_ids)} SNPs. max_snp_count = {max_snp_count}')
    
    return selection


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        # for isolated testing
        Snakemake = namedtuple('Snakemake', ['input', 'output', 'params', 'resources', 'log'])
        snakemake = Snakemake(
            input={'gwases': ['test.input']},
            output={'name': 'test.output'},
            params={'parameter': 'value'},
            resources={'resource_name': 'value'},
            log=['test.log']
        )

    logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

    gwas_sources = snakemake.input['gwases']
    snplist_path = snakemake.output['snplist']
    max_snp_count = int(snakemake.wildcards['snp_count'])

    gwas_data = read_all_gwas(gwas_sources)
    selection = _get_selection_snp_ids(gwas_data, max_snp_count)
    print(f'topk strategy: final selection of SNPs contains {len(selection)} SNPs')

    with open(snplist_path, 'w') as file:
        file.write('\n'.join(selection))