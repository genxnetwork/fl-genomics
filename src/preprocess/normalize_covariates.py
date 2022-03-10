from collections import namedtuple
import logging
import pandas
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        # for isolated testing
        Snakemake = namedtuple('Snakemake', ['input', 'output', 'params', 'resources', 'log'])
        snakemake = Snakemake(
            input={'name': 'test.input'},
            output={'name': 'test.output'},
            params={'parameter': 'value'},
            resources={'resource_name': 'value'},
            log=['test.log']
        )

    logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

    covariates = pandas.read_table(snakemake.input['train'])
    to_normalize = snakemake.params['to_normalize']
    output_path = snakemake.output['train']
    logging.info(f'We will normalize {to_normalize} covariates from {covariates.columns}')

    scaler = StandardScaler()
    covariates.loc[:, to_normalize] = scaler.fit_transform(covariates.loc[:, to_normalize])
    covariates.to_csv(output_path, sep='\t', index=False)