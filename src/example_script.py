from collections import namedtuple
import logging


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

    input = snakemake.input[0]
    named_input = snakemake.input['name']
    output = snakemake.output[0]
    named_output = snakemake.output['name']
    parameter = snakemake.params['parameter']
    resource = snakemake.resources['resource_name']
    log_file = snakemake.log[0]

    # write something to output