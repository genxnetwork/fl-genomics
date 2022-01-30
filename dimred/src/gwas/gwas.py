import os
import hydra
from omegaconf import DictConfig
from utils.plink import run_plink
import shutil


def get_gwas_output_path(output_path: str, phenotype_name: str, phenotype_type: str):
    if phenotype_type == 'binary':
        return f'{output_path}.{phenotype_name}.glm.logistic.firth'
    else:
        return f'{output_path}.{phenotype_name}.glm.linear'


@hydra.main(config_path='configs', config_name='gwas')
def main(cfg: DictConfig):
    os.makedirs(os.path.join(cfg.split_dir, 'gwas'), exist_ok=True)
    args = [
        '--pfile', cfg.genotype.train,
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
        f'{cfg.output.path}.{cfg.phenotype.name}.tsv')


if __name__ == '__main__':
    main()