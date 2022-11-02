import os
import gc

import numpy as np
import pandas as pd
import scipy.sparse.linalg as linalg

from utils.plink import run_plink


class FederatedPCASimulationRunner:
    """
    Strategy:
    ---------

    1. Each node has serveral fold subsets: train, test, validation.
    2. Federated PCA is computed using only train subset of each node.
    3. Result projection is applied to all three subsets: train, test, validation.
    """

    # Results of federated PCA aggegation are stored for the <node identifier = ALL> on the filesystem
    ALL = 'ALL'

    def __init__(
        self,
        source_folder,
        result_folder,
        variant_ids_file,
        n_components=2,
        method='P-STACK',
        nodes=['AFR', 'SAS', 'EUR', 'AMR', 'EAS'],
        folds_number=10,
        train_foldname_template='fold_%i_train',
        test_foldname_template='fold_%i_test',
        validation_foldname_template='fold_%i_val'
    ):
        """
        Two methods are available: P-COV and P-STACK. Both of them provide the same results but for the case
        when number of features >> number of samples, P-STACK's approach is more memory-efficient..
        """

        self.source_folder = source_folder
        self.result_folder = result_folder
        self.variant_ids_file = variant_ids_file
        self.n_components = n_components
        self.method = method
        self.nodes = list(nodes)
        self.folds_number = folds_number
        self.train_foldname_template = train_foldname_template
        self.test_foldname_template = test_foldname_template
        self.validation_foldname_template = validation_foldname_template
        self.parts = [
            self.train_foldname_template,
            self.test_foldname_template,
            self.validation_foldname_template
        ]

        # TODO: move this outside of the constructor since directory creation is weird to be here.
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

        for node in self.nodes + [self.ALL]:
            if not os.path.isdir(os.path.join(result_folder, node)):
                os.mkdir(os.path.join(result_folder, node))

    def compute_allele_frequencies(self):
        """
        Compute centralized allele frequencies by joining plink *.acount files
        obtained separately for each node.
        """

        for node in self.nodes:
            for fold in range(self.folds_number):
                pfile = os.path.join(self.source_folder, node, self.train_foldname_template % fold)
                output = os.path.join(self.result_folder, node, self.train_foldname_template % fold)

                run_plink(args_list=[
                    '--pfile', pfile,
                    '--extract', self.variant_ids_file,
                    '--freq', 'counts',
                    '--out', output
                ])

        # Join allele frequency files for each fold
        for fold in range(self.folds_number):
            acount_data_list = []
            for node in self.nodes:
                acount_file = os.path.join(
                    self.result_folder, node, self.train_foldname_template % fold + '.acount'
                )

                acount_data_list.append(pd.read_csv(acount_file, sep='\t', header=0))

            result = acount_data_list[0]
            for acount_data in acount_data_list[1:]:
                # IDs consistency check before merge
                if not np.all(result['ID'] == acount_data['ID']):
                    raise ValueError('Variant IDs are not consistent between *.acount plink files')

                result['ALT_CTS'] += acount_data['ALT_CTS']
                result['OBS_CT'] += acount_data['OBS_CT']

            output_file = os.path.join(
                self.result_folder, self.ALL, self.train_foldname_template % fold + '.acount'
            )

            result.to_csv(output_file, sep='\t', index=False, header=True)

    def run(self):
        self.compute_allele_frequencies()

        for fold in range(self.folds_number):
            for node in self.nodes:
                self.run_client_pca(node, fold)

            # Aggregate client results
            self.run_server_aggregation(fold)

            for node in self.nodes + ['ALL']:
                self.run_client_projection(node, fold)

    def run_client_pca(self, node, fold):
        """
        Performs local PCA with plink
        """

        client_pfile = os.path.join(self.source_folder, node, self.train_foldname_template % fold)
        output_pca_file = os.path.join(self.result_folder, node, self.train_foldname_template % fold)
        allele_frequencies_file = os.path.join(
            self.result_folder, self.ALL, self.train_foldname_template % fold + '.acount'
        )

        n_samples = len(pd.read_csv(client_pfile + '.psam', sep='\t', header=0))
        run_plink(args_list=[
            '--pfile', client_pfile,
            '--extract', self.variant_ids_file,
            '--read-freq', allele_frequencies_file,
            '--pca', 'allele-wts', str(n_samples - 1),
            '--out', output_pca_file,
        ])

    def run_server_aggregation(self, fold):
        if self.method == 'P-STACK':
            aggregate, ids = self.load_pstack_component(self.nodes[0], fold)
            for node in self.nodes[1:]:
                component, other_ids = self.load_pstack_component(node, fold)

                # IDs consistency check before merge
                if not np.all(ids == other_ids):
                    raise ValueError('Variant IDs are not consistent between *.eigenvec.allele plink files')

                aggregate = np.concatenate([aggregate, component], axis=0)

        elif self.method == 'P-COV':
            aggregate = self.load_pcov_component(self.nodes[0], fold)
            for node in self.nodes[1:]:
                component = self.load_pcov_component(node, fold)
                aggregate = aggregate + component

        del component
        gc.collect()

        _, _, evectors = linalg.svds(aggregate, k=self.n_components)

        del aggregate
        gc.collect()

        # Flip eigenvectors since 'linalg.svds' returns them in a reversed order
        evectors = np.flip(evectors, axis=0)

        self.create_plink_eigenvec_allele_file(fold, evectors)

    def load_pstack_component(self, node, fold):
        evectors, evalues, ids = self.read_pca_results(node, fold)
        evalues_matrix = np.zeros((len(evalues), len(evalues)))
        np.fill_diagonal(evalues_matrix, evalues)

        return np.dot(np.sqrt(evalues_matrix), evectors.T), ids

    def load_pcov_component(self, node, fold):
        evectors, evalues, _ = self.read_pca_results(node, fold)
        evalues_matrix = np.zeros((len(evalues), len(evalues)))
        np.fill_diagonal(evalues_matrix, evalues)

        return np.dot(np.dot(evectors, evalues_matrix), evectors.T)

    def read_pca_results(self, node, fold):
        """
        Read plink results into NumPy arrays.
        """

        evalues_file = os.path.join(
            self.result_folder, node, self.train_foldname_template % fold + '.eigenval'
        )

        evectors_file = os.path.join(
            self.result_folder, node, self.train_foldname_template % fold + '.eigenvec.allele'
        )

        evalues = pd.read_csv(evalues_file, sep='\t', header=None)
        evectors = pd.read_csv(evectors_file, sep='\t', header=0)

        return evectors[evectors.columns[5:]].to_numpy(), evalues[0].to_numpy(), evectors['ID'].to_numpy()

    def create_plink_eigenvec_allele_file(self, fold, evectors):
        # Take the first 5 columns from one of the nodes to mimic combined plink .eigenvec.allele file
        client_allele_file = os.path.join(
            self.result_folder, self.nodes[0], self.train_foldname_template % fold + '.eigenvec.allele'
        )

        client_allele = pd.read_csv(client_allele_file, sep='\t', header=0)
        server_allele = client_allele[client_allele.columns[0:5]]

        for n in range(self.n_components):
            '''
            FIXME: Temporal solution

            Motivation
            ----------

            Principal components obtained from `svds` are normalized, i.e. have the norm the equals 1.
            Due to the normalization, resulting projection have small values for sample coordinates
            in the principal components space. Our classifier cannot learn from such values, since
            they are too small and we do not have a data normalization step.

            Normalization
            -------------

            Plink principal components are normalized in a different way. We use one of the client files
            to compute principal components norm and then renormalize the federated principal components
            in the same way to mimic plink PCA scale.

            This should be considered as a temporal solution. In the future data normalization may be
            added into the data loading stage, it can be implemented for our federated case as well.
            '''

            component_name = f'PC{n + 1}'
            current_norm = np.linalg.norm(evectors[n])
            plink_norm = np.linalg.norm(client_allele.iloc[:, n + 5])
            server_allele[component_name] = evectors[n] * plink_norm / current_norm

        server_allele_file = os.path.join(
            self.result_folder, self.ALL, self.train_foldname_template % fold + '.eigenvec.allele'
        )

        server_allele.to_csv(server_allele_file, sep='\t', header=True, index=False)

    def run_client_projection(self, node, fold):
        allele_frequencies_file = os.path.join(
            self.result_folder, self.ALL, self.train_foldname_template % fold + '.acount'
        )

        server_allele_file = os.path.join(
            self.result_folder, self.ALL, self.train_foldname_template % fold + '.eigenvec.allele'
        )

        for part in self.parts:
            pfile = os.path.join(self.source_folder, node, part % fold)
            sscore_file = os.path.join(self.result_folder, node, part % fold + '_projections.csv.eigenvec')

            run_plink(args_list=[
                '--pfile', pfile,
                '--extract', self.variant_ids_file,
                '--read-freq', allele_frequencies_file,
                '--score', server_allele_file, '2', '5',
                '--score-col-nums', f'6-{6 + self.n_components - 1}',
                '--out', sscore_file
            ])
