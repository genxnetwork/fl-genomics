from typing import List, Set, Tuple
import hydra
from omegaconf import DictConfig
import pandas
import os
from utils.plink import run_plink
import matplotlib.pyplot as plt


def _read_all_gwas(gwas_dir: str, phenotype_name: str, node_count: int, fold_index: int) -> List[pandas.DataFrame]:
    """
    Reads all GWAS results for a particular phenotype and split from {gwas_dir}

    Args:
        gwas_dir (str): Dir with table-delimited files with GWAS results from plink 2.0 
        phenotype_name (str): Name of the phenotype
        node_count (int): Number of nodes in split

    Returns:
        List[pandas.DataFrame]: List of GWAS results with #CHROM, POS, LOG10_P columns and ID as index 
    """    
    results = []
    for node_index in range(node_count):
        gwas_path = os.path.join(gwas_dir, phenotype_name, f'node_{node_index}', f'fold_{fold_index}.tsv')
        gwas = pandas.read_table(gwas_path)
        results.append(gwas.loc[:, ['#CHROM', 'POS', 'ID', 'LOG10_P']].set_index('ID'))
    return results


def _get_topk_snps(gwas: pandas.DataFrame, max_snp_count: int) -> pandas.DataFrame:
    sorted_gwas = gwas.sort_values(by='LOG10_P', axis='index', ascending=False).iloc[:max_snp_count, :]
    return sorted_gwas.drop('LOG10_P', axis='columns')


def _get_snp_list_path(strategy: str, split_dir: str) -> str:
    snplists_dir = os.path.join(split_dir, 'gwas', 'snplists')
    os.makedirs(snplists_dir, exist_ok=True)
    return os.path.join(snplists_dir, f'union_{strategy}.snplist')


def _get_pfile_dir(split_dir: str, split_index: int, strategy: str) -> str:
    return os.path.join(split_dir, 'genotypes', 'union', f'split_{split_index}', strategy)


def _get_selection_snp_ids(gwas_data: List[pandas.DataFrame], max_snp_count: int) -> Set:
    intersection_ids = set(gwas_data[0].index)
    for gwas in gwas_data[1:]:
        intersection_ids &= set(gwas.index)

    topk_data = [_get_topk_snps(gwas, max_snp_count) for gwas in gwas_data]
        
    union = pandas.concat(topk_data, join='inner', axis='index')
    union_ids = set(union.index)

    # union.sort_values(by=['#CHROM', 'POS'], inplace=True, ascending=True)
    selection = intersection_ids & union_ids
    
    print(f'intersection of SNPs before GWAS for all datasets has {len(intersection_ids)} SNPs. max_snp_count = {max_snp_count}')
    print(f'topk strategy: union of SNPs contains {len(union_ids)} SNPs. max_snp_count = {max_snp_count}')
    
    return selection


def topk(cfg: DictConfig):
    """
    Selects {cfg.max_snp_count} most significant SNPs from each node and writes a union into file in {cfg.split_dir}
    
    Args:
        cfg (DictConfig): Main script config
    """    
    gwas_data = _read_all_gwas(os.path.join(cfg.split_dir, 'gwas'), cfg.phenotype.name, cfg.split_count)
    
    selection = _get_selection_snp_ids(gwas_data, cfg.max_snp_count)

    print(f'topk strategy: final selection of SNPs contains {len(selection)} SNPs')

    snp_list_path = _get_snp_list_path(cfg.strategy, cfg.split_dir)
    with open(snp_list_path, 'w') as file:
        file.write('\n'.join(selection))


def generate_pfiles(cfg: DictConfig):
    """Generates pfiles with the same set of SNPs for each node

    Args:
        cfg (DictConfig): Main script config
    """    
    snp_list_path = _get_snp_list_path(cfg.strategy, cfg.split_dir)
    for node_index in range(cfg.split_count):
        pfile_dir = _get_pfile_dir(cfg.split_dir, node_index, cfg.strategy)
        os.makedirs(pfile_dir, exist_ok=True)

        for part in ['train', 'val']:
            
            phenotype_path = os.path.join(cfg.split_dir, 'only_phenotypes', f'{cfg.phenotype.name}_split_{split_index}_{part}.tsv')
            output_path = os.path.join(pfile_dir, f'{cfg.phenotype.name}.{part}_top{cfg.max_snp_count}')
            pfile_path = os.path.join(cfg.split_dir, 'genotypes', f'split_{split_index}_filtered_{part}')
            
            args = ['--pfile', pfile_path,
                    '--extract', snp_list_path, '--keep', phenotype_path, 
                    '--make-pgen', '--out', output_path]
            run_plink(args)
        
        print(f'Top {cfg.max_snp_count} SNP selection completed for split {split_index} and part {part} with strategy {cfg.strategy}')


def plot_union_thresholds(cfg: DictConfig):
    """Plots length of SNP union for each snp_count in {cfg.analysis.snp_counts} into {cfg.analysis.plot_path}

    Args:
        cfg (DictConfig): Main script config
    """    
    gwas_data = _read_all_gwas(os.path.join(cfg.split_dir, 'gwas'), cfg.phenotype.name, cfg.split_count)

    sel_lens = []
    for threshold in cfg.analysis.snp_counts:
        selection = _get_selection_snp_ids(gwas_data, threshold)
        sel_lens.append(len(selection))

    plt.figure(figsize=(15, 12))
    plt.grid(which='both')
    plt.title('Common SNPs between nodes after GWAS', fontsize=28)
    plt.xlabel('Number of SNPs taken from each node', fontsize=20)
    plt.ylabel('Number of SNPs in resulting dataset', fontsize=20)
    plt.plot(cfg.analysis.snp_counts, sel_lens, linewidth=2, marker='o', markersize=16)
    plt.xticks(cfg.analysis.snp_counts, [str(t) for t in cfg.analysis.snp_counts], fontsize=20, rotation=90, visible=True)
    plt.savefig(cfg.analysis.plot_path)


def pvalue(cfg: DictConfig):
    pass


def mixed(cfg: DictConfig):
    pass


@hydra.main(config_path='configs', config_name='union')
def main(cfg: DictConfig):
    # select SNPs based on GWAS results with specified union strategy
    strategy = {
        'topk': topk,
        'pvalue': pvalue,
        'mixed': mixed,
    }[cfg.strategy]
    strategy(cfg)
    # plot lengths of unions for different thresholds
    plot_union_thresholds(cfg)
    # generate pfiles for each node
    generate_pfiles(cfg)
    

if __name__ == '__main__':
    main()