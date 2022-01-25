import hydra
from omegaconf import DictConfig
from utils.plink import run_plink
import shutil


def get_gwas_output_path(output_path: str, phenotype_type: str):
    if phenotype_type == 'binary':
        return output_path + '.glm.logistic.firth.tsv'
    else:
        return output_path + '.glm.linear.tsv'




@hydra.main(config_path='configs', config_name='gwas')
def main(cfg: DictConfig):
    args = [
        '--pfile', cfg.genotype.train,
        '--covar', cfg.covariates.path,
        '--pheno', cfg.phenotype.path,
        '--glm', 'sex', 'log10', 'hide-covar',
        '--out', cfg.output.path,
        '--threads', cfg.threads
    ]
    run_plink(args)

    # plink GWAS output filename depends on phenotype and regression. 
    # We rename it to be always cfg.output.path + .tsv 
    shutil.move(get_gwas_output_path(cfg.output.path, cfg.phenotype.type), cfg.output.path + '.tsv')


if __name__ == '__main__':
    main()