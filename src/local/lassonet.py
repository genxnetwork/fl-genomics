from collections import namedtuple
import logging


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        # for isolated testing
        Snakemake = namedtuple('Snakemake', ['input', 'output', 'params'])
        snakemake = Snakemake(
            input={'genotype': 'test.input'},
            output={'results': 'test.output'},
            params={'pfile': 'test'}
        )

    named_input = snakemake.input['genotype']
    named_output = snakemake.output['results']
    pfile = snakemake.params['pfile']
    print(named_input)
    print(named_output)
    print(pfile)
    # write something to output