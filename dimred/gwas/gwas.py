from email.policy import default
import subprocess
import hydra
from omegaconf import DictConfig
from typing import List
import shutil


def run_plink(args: List[str]):
    plink = subprocess.run(['plink2'] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if plink.returncode != 0:
        raise RuntimeError(plink.stderr.decode('utf-8'))
    

def get_gwas_output_path(output_path: str, phenotype_type: str):
    if phenotype_type == 'binary':
        return output_path + '.glm.logistic.firth.tsv'
    else:
        return output_path + '.glm.linear.tsv'


@hydra.main(config_path='configs', config_name=default)
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

    shutil.move(get_gwas_output_path(cfg.output.path, cfg.phenotype.type), cfg.output.path + '.tsv')


if __name__ == '__main__':
    main()