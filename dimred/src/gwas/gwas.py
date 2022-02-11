import os
import hydra
from omegaconf import DictConfig
from utils.plink import run_plink, get_gwas_output_path
import shutil


@hydra.main(config_path='configs', config_name='gwas')
def main(cfg: DictConfig):
    os.makedirs(cfg.output.root_dir, exist_ok=True)
    args = [
        '--pfile', cfg.genotype,
        '--covar', cfg.covariates.path,
        '--pheno', cfg.phenotype.path,
        '--no-psam-pheno', '--pheno-name', cfg.phenotype.name,
        '--glm', 'no-x-sex', 'log10', 'hide-covar',
        '--out', cfg.output.path,
        '--threads', str(cfg.threads)
    ]
    run_plink(args)

    # plink GWAS output filename depends on phenotype and regression. 
    # We rename it to be always cfg.output.path + .tsv 
    shutil.move(get_gwas_output_path(cfg.output.path, cfg.phenotype.name, cfg.phenotype.type), 
        f'{cfg.output.path}.tsv')

    print(f'GWAS for phenotype {cfg.phenotype.name} and split {cfg.node_index} and fold {cfg.fold_index} finished ')


if __name__ == '__main__':
    main()