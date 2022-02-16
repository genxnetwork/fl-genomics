from typing import List, Set, Tuple
import hydra
from omegaconf import DictConfig
import pandas
import os
from gwas.train_val_split import FOLD_COUNT
from utils.split import Split
from utils.plink import run_plink
import matplotlib.pyplot as plt


def _read_all_gwas(split: Split, node_count: int, fold_index: int) -> List[pandas.DataFrame]:
    """
    Reads all GWAS results for a particular phenotype and split from {gwas_dir}

    Args:
        split (str): Object for paths manipulation 
        node_count (int): Number of nodes in split

    Returns:
        List[pandas.DataFrame]: List of GWAS results with #CHROM, POS, LOG10_P columns and ID as index 
    """    
    results = []
    for node_index in range(node_count):
        gwas_path = split.get_gwas_path(node_index, fold_index, adjusted=True)
        gwas = pandas.read_table(gwas_path)
        results.append(gwas.loc[:, ['#CHROM', 'POS', 'ID', 'LOG10_P']].set_index('ID'))
    return results


def _get_topk_snps(gwas: pandas.DataFrame, max_snp_count: int) -> pandas.DataFrame:
    sorted_gwas = gwas.sort_values(by='LOG10_P', axis='index', ascending=False).iloc[:max_snp_count, :]
    return sorted_gwas.drop('LOG10_P', axis='columns')


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


def topk(cfg: DictConfig, split: Split):
    """
    Selects {cfg.max_snp_count} most significant SNPs from each node and writes a union into file in {cfg.split_dir}
    
    Args:
        cfg (DictConfig): Main script config
    """    
    gwas_data = _read_all_gwas(split, cfg.node_count, cfg.fold_index)
    
    selection = _get_selection_snp_ids(gwas_data, cfg.max_snp_count)

    print(f'topk strategy: final selection of SNPs contains {len(selection)} SNPs')

    for node_index in range(cfg.node_count):
        snp_list_path = split.get_snplist_path(cfg.strategy, node_index, cfg.fold_index)
        print(f'We have {len(selection)} SNPs to write to {snp_list_path}')
        with open(snp_list_path, 'w') as file:
            file.write('\n'.join(selection))


def generate_pfiles(cfg: DictConfig, split: Split):
    """Generates pfiles with the same set of SNPs for each node

    Args:
        cfg (DictConfig): Main script config
    """    
    for node_index in range(cfg.node_count):

        for part in ['train', 'val', 'test']:
            
            phenotype_path = split.get_phenotype_path(node_index, cfg.fold_index, part)
            
            snplist_path = split.get_snplist_path(cfg.strategy, node_index, cfg.fold_index)
            output_path = split.get_topk_pfile_path(cfg.strategy, node_index, cfg.fold_index, cfg.max_snp_count, part, None)
            pfile_path = split.get_source_pfile_path(node_index)
            print(phenotype_path)
            args = ['--pfile', pfile_path,
                    '--extract', snplist_path, '--keep', phenotype_path, 
                    '--make-pgen', '--out', output_path]
            run_plink(args)
        
        print(f'Top {cfg.max_snp_count} SNP selection completed for node {node_index}, fold {cfg.fold_index} and part {part} with strategy {cfg.strategy}')


def plot_union_thresholds(cfg: DictConfig, split: Split):
    """Plots length of SNP union for each snp_count in {cfg.analysis.snp_counts} into {cfg.analysis.plot_path}

    Args:
        cfg (DictConfig): Main script config
    """    
    gwas_data = _read_all_gwas(split, cfg.node_count, cfg.fold_index)

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
    print(f'Saving fig into {cfg.analysis.plot_path}')
    plt.savefig(cfg.analysis.plot_path)


def pvalue(cfg: DictConfig, split: Split):
    pass


def mixed(cfg: DictConfig, split: Split):
    pass


def no_union(cfg: DictConfig, split: Split):
    """Selects top k SNPs by p-value from each node independently. 
    Each node will have DIFFERENT set of SNPs

    Args:
        cfg (DictConfig): Main script config
    """    
    gwas_data = _read_all_gwas(split, cfg.node_count, cfg.fold_index)
    topk_data = [_get_topk_snps(gwas, cfg.max_snp_count) for gwas in gwas_data]

    for node_index, topk in enumerate(topk_data):
        snp_list_path = split.get_snplist_path(cfg.strategy, node_index, cfg.fold_index)
        with open(snp_list_path, 'w') as file:
            file.write('\n'.join(topk.index.tolist()))


@hydra.main(config_path='configs', config_name='union')
def main(cfg: DictConfig):
    # select SNPs based on GWAS results with specified union strategy
    strategy = {
        'topk': topk,
        'pvalue': pvalue,
        'mixed': mixed,
        'no_union': no_union
    }[cfg.strategy]
    split = Split(cfg.split_dir, cfg.phenotype.name, cfg.node_count, FOLD_COUNT)
    strategy(cfg, split)
    # plot lengths of unions for different thresholds
    if cfg.strategy == 'topk':
        print(f'Plotting union threshold')
        plot_union_thresholds(cfg, split)
    # generate pfiles for each node
    generate_pfiles(cfg, split)
    

if __name__ == '__main__':
    main()