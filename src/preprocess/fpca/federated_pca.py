import os
import random
import subprocess

import numpy as np
import pandas as pd
import scipy.sparse.linalg as linalg


SOURCE_FOLDER = '/media/storage/TG/data/chip/superpop_split/genotypes'
RESULT_FOLDER = '/home/genxadmin/federated-pca/data'
VARIANT_IDS_FOLDER = RESULT_FOLDER + '/ids'
PRUNED_PLINK_FILES_FOLDER = RESULT_FOLDER + '/pruned'
NODES = ['AFR', 'SAS', 'EUR', 'AMR', 'EAS']
ALL = 'ALL'
PCA_FOLDER = RESULT_FOLDER + '/pca'


def init():
    if not os.path.isdir(RESULT_FOLDER):
        os.mkdir(RESULT_FOLDER)

    if not os.path.isdir(VARIANT_IDS_FOLDER):
        os.mkdir(VARIANT_IDS_FOLDER)

    if not os.path.isdir(PRUNED_PLINK_FILES_FOLDER):
        os.mkdir(PRUNED_PLINK_FILES_FOLDER)

    if not os.path.isdir(PCA_FOLDER):
        os.mkdir(PCA_FOLDER)


def prune(portion=0.001):
    """
    Imitates variants pruning using plink. Creates a list of pruned ids.
    """

    input_pfile = os.path.join(SOURCE_FOLDER, ALL + '_filtered')
    output_ids_file = os.path.join(VARIANT_IDS_FOLDER, 'pruned.ids')

    # Get variants list from .pvar plink file
    variants = pd.read_csv(input_pfile + '.pvar', sep='\t', header=None, comment='#')
    n_variants = len(variants)

    random.seed(42)
    ids = random.sample(list(range(n_variants)), round(portion * n_variants))
    ids.sort()

    variants_ids = variants[2][ids]
    variants_ids = variants_ids[variants_ids != '.']

    variants_ids.to_csv(output_ids_file, sep='\t', index=False, header=False)


def compute_allele_frequencies(node, variant_ids_file):
    input_pfile = os.path.join(SOURCE_FOLDER, node + '_filtered')
    output_file = os.path.join(PCA_FOLDER, node)

    subprocess.run([
        'plink2', '--pfile', input_pfile,
        '--extract', variant_ids_file,
        '--freq', 'counts',
        '--out', output_file
    ])


def client(node, variant_ids_file, allele_frequencies_file=None):
    """
    Performs local PCA with plink
    """

    input_pfile = os.path.join(SOURCE_FOLDER, node + '_filtered')
    output_pca_file = os.path.join(PCA_FOLDER, node)

    n_samples = len(pd.read_csv(input_pfile + '.psam', sep='\t', header=0))
    command = [
        'plink2', '--pfile', input_pfile,
        '--extract', variant_ids_file,
        '--pca', 'allele-wts', str(n_samples - 1),
        '--out', output_pca_file
    ]

    if allele_frequencies_file is not None:
        command += ['--read-freq', allele_frequencies_file]
    else:
        command += ['--freq', 'counts']

    # Run plink
    subprocess.run(command)


def server(nodes, n_components=2, method='AP-STACK'):
    if method == 'AP-STACK':
        aggregate = load_pstack_component(nodes[0])
        for node in nodes[1:]:
            component = load_pstack_component(node)
            aggregate = np.concatenate([aggregate, component], axis=0)

    elif method == 'P-COV':
        aggregate = load_pcov_component(nodes[0])
        for node in nodes[1:]:
            component = load_pcov_component(node)
            aggregate = aggregate + component

    _, evalues, evectors = linalg.svds(aggregate, k=n_components)

    # FIXME: revert evectors and evalues
    return evectors, evalues


def load_pstack_component(node):
    evectors, evalues = read_pca_results(node)
    evalues_matrix = np.zeros((len(evalues), len(evalues)))
    np.fill_diagonal(evalues_matrix, evalues)

    return np.dot(np.sqrt(evalues_matrix), evectors.T)


def load_pcov_component(node):
    evectors, evalues = read_pca_results(node)
    evalues_matrix = np.zeros((len(evalues), len(evalues)))
    np.fill_diagonal(evalues_matrix, evalues)

    return np.dot(np.dot(evectors, evalues_matrix), evectors.T)


def read_pca_results(node):
    """
    Read plink results into NumPy arrays.
    """

    evalues_file = os.path.join(PCA_FOLDER, node + '.eigenval')
    evectors_file = os.path.join(PCA_FOLDER, node + '.eigenvec.allele')

    evalues = pd.read_csv(evalues_file, sep='\t', header=None)
    evectors = pd.read_csv(evectors_file, sep='\t', header=0)

    return evectors[evectors.columns[5:]].to_numpy(), evalues[0].to_numpy()


def create_plink_eigenvec_allele_file(evectors, n_components=2):
    # Take the first 5 columns from one of the nodes to mimic combined plink .eigenvec.allele file
    client_allele_file = os.path.join(PCA_FOLDER, NODES[0] + '.eigenvec.allele')
    allele = pd.read_csv(client_allele_file, sep='\t', header=0)
    allele = allele[allele.columns[0:5]]

    for n in range(n_components):
        component_name = f'PC{n + 1}'
        allele[component_name] = evectors[n]

    server_allele_file = os.path.join(PCA_FOLDER, 'federated.eigenvec.allele')
    allele.to_csv(server_allele_file, sep='\t', header=True, index=False)


def client_projection(node, variant_ids_file, allele_frequencies_file, plink_allele_projector, n_components=2):
    input_pfile = os.path.join(SOURCE_FOLDER, node + '_filtered')
    output_pca_file = os.path.join(PCA_FOLDER, node + '.projection')

    subprocess.run([
        'plink2', '--pfile', input_pfile,
        '--extract', variant_ids_file,
        '--read-freq', allele_frequencies_file,
        '--score', plink_allele_projector, '2', '5',
        '--score-col-nums', f'6-{6 + n_components - 1}',
        '--out', output_pca_file
    ])


def read_evectors_from_sscore(node):
    input_sscore_file = os.path.join(PCA_FOLDER, node + '.projection.sscore')
    sscore = pd.read_csv(input_sscore_file, sep='\t', header=0)
    sscore = sscore[sscore.columns[3:]]
    return sscore
