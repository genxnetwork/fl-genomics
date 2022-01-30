from typing import List, Tuple
import hydra
from omegaconf import DictConfig
import pandas
import os
from utils.plink import run_plink


def _read_all_gwas(gwas_dir: str, phenotype_name: str, split_count: int) -> List[pandas.DataFrame]:
    results = []
    for split_index in range(split_count):
        gwas_path = os.path.join(gwas_dir, f'split{split_index}.{phenotype_name}.tsv')
        gwas = pandas.read_table(gwas_path)
        results.append(gwas.loc[:, ['#CHROM', 'POS', 'ID', 'LOG10_P']].set_index('ID'))
    return results


def _get_topk_snps(gwas: pandas.DataFrame, max_snp_count: int) -> pandas.DataFrame:
    sorted_gwas = gwas.sort_values(by='LOG10_P', axis='index', ascending=False).iloc[:max_snp_count, :]
    return sorted_gwas.drop('LOG10_P', axis='columns')


def _get_snp_list_path(strategy: str, split_dir: str) -> str:
    return os.path.join(split_dir, 'gwas', f'union_{strategy}.snplist')


def _get_pfile_dir(split_dir: str, split_index: int, strategy: str) -> str:
    return os.path.join(split_dir, f'split{split_index}', strategy)


def topk(cfg: DictConfig):
    gwas_data = _read_all_gwas(os.path.join(cfg.split_dir, 'gwas'), cfg.phenotype.name, cfg.split_count)
    
    intersection_ids = set(gwas_data[0].index)
    for gwas in gwas_data[1:]:
        intersection_ids &= set(gwas.index)

    topk_data = [_get_topk_snps(gwas, cfg.max_snp_count) for gwas in gwas_data]
        
    union = pandas.concat(topk_data, join='inner', axis='index')
    union_ids = set(union.index)

    # union.sort_values(by=['#CHROM', 'POS'], inplace=True, ascending=True)
    selection = intersection_ids & union_ids
    print(f'intersection of SNPs before GWAS for all datasets has {len(intersection_ids)} SNPs')
    print(f'topk strategy: union of SNPs contains {len(union_ids)} SNPs')
    print(f'topk strategy: final selection of SNPs contains {len(selection)} SNPs')

    snp_list_path = _get_snp_list_path(cfg.strategy, cfg.split_dir)
    with open(snp_list_path, 'w') as file:
        file.write('\n'.join(selection))


def generate_pfiles(cfg: DictConfig):
    snp_list_path = _get_snp_list_path(cfg.strategy, cfg.split_dir)
    for split_index in range(cfg.split_count):
        pfile_dir = _get_pfile_dir(cfg.split_dir, split_index, cfg.strategy)
        os.makedirs(pfile_dir, exist_ok=True)
        for part in ['train', 'val']:
            args = ['--pfile', os.path.join(cfg.split_dir, 'genotypes', f'split{split_index}_filtered_{part}'),
                    '--extract', snp_list_path,
                    '--make-pgen', '--out', os.path.join(pfile_dir, f'train_top{cfg.max_snp_count}')]
            run_plink(args)
        
        print(f'Top {cfg.max_snp_count} SNP selection completed for split {split_index} and part {part} with strategy {cfg.strategy}')


def pvalue(cfg: DictConfig):
    pass


def mixed(cfg: DictConfig):
    pass


@hydra.main(config_path='configs', config_name='union')
def main(cfg: DictConfig):
    strategy = {
        'topk': topk,
        'pvalue': pvalue,
        'mixed': mixed,
    }[cfg.strategy]
    strategy(cfg)
    generate_pfiles(cfg)


if __name__ == '__main__':
    main()