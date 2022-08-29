import os
import sys
import logging
import argparse
import pandas as pd

from preprocess.qc import QC
from preprocess.splitter_tg import SplitTGHeter
from utils.plink import run_plink
from utils.split import Split
from preprocess.train_val_split import CVSplitter
from configs.global_config import TG_BFILE_PATH, SPLIT_DIR, SPLIT_GENO_DIR, FEDERATED_PCA_DIR
from configs.qc_config import sample_qc_config, variant_qc_config
from configs.split_config import TG_SUPERPOP_DICT, FOLDS_NUMBER
from configs.pruning_config import default_pruning

from preprocess.pruning import PlinkPruningRunner
from preprocess.federated_pca import FederatedPCASimulationRunner


class Stage:
    VARIANT_QC = 'variant-qc'
    POPULATION_SPLIT = 'population-split'
    SAMPLE_QC = 'sample-qc'
    PRUNING = 'pruning'
    FOLD_SPLIT = 'fold-split'
    FEDERATED_PCA = 'federated-pca'
    CENTRALIZED_PCA = 'pca'

    @classmethod
    def all(cls):
        return {
            cls.VARIANT_QC, cls.POPULATION_SPLIT, cls.SAMPLE_QC,
            cls.PRUNING, cls.FOLD_SPLIT, cls.FEDERATED_PCA
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'stages', metavar='stage', type=str, nargs='+',
        help=f'Available stages: {Stage.all()}'
    )
    args = parser.parse_args()
    stages = set(args.stages) if args.stages else Stage.all()


    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger()


    # # Generate file with sample IDs that pass central QC with plink
    # logger.info(f'Running sample QC and saving valid ids to {TG_SAMPLE_QC_IDS_PATH}')
    # sample_qc(bin_file_path=TG_BFILE_PATH, output_path=TG_SAMPLE_QC_IDS_PATH, bin_file_type='--pfile')
    #
    # logger.info(f'Running global PCA')
    # os.makedirs(os.path.join(TG_DATA_ROOT, 'pca'), exist_ok=True)
    # PCA().run(input_prefix=TG_BFILE_PATH, pca_config=pca_config_tg,
    #           output_path=os.path.join(TG_DATA_ROOT, 'pca', 'global'),
    #           scatter_plot_path=None,
    #           # scatter_plot_path=os.path.join(TG_OUT, 'global_pca.html'),
    #           bin_file_type='--bfile')


    # 1. Centralised variant QC
    varqc_prefix = TG_BFILE_PATH + '_varqc'
    if Stage.VARIANT_QC in stages:
        logger.info('Centralised variant QC')
        QC.qc(input_prefix=TG_BFILE_PATH, output_prefix=varqc_prefix, qc_config=variant_qc_config)

    # 2. Split into ethnic datasets and then QC each local dataset
    nodes = set(TG_SUPERPOP_DICT.values())
    if Stage.POPULATION_SPLIT in stages:
        logger.info('Splitting ethnic dataset')
        SplitTGHeter().split(input_prefix=varqc_prefix, make_pgen=True)

    # 3. Perform sample QC on each node separately
    if Stage.SAMPLE_QC in stages:
        for local_prefix in nodes:
            logger.info(f'Running local QC for {local_prefix}')
            local_samqc_prefix = os.path.join(SPLIT_GENO_DIR, local_prefix) + '_filtered'
            QC.qc(
                input_prefix=os.path.join(SPLIT_GENO_DIR, local_prefix),
                output_prefix=local_samqc_prefix,
                qc_config=sample_qc_config
            )

    # 4. Perform pruning for each node separately
    if Stage.PRUNING in stages:
        logger.info(f'Pruning with plink')
        PlinkPruningRunner(
            source_directory=SPLIT_GENO_DIR,
            nodes=nodes,
            result_filepath=os.path.join(SPLIT_GENO_DIR, 'ALL.prune.in'),
            node_filename_template='%s_filtered'
        ).run(**default_pruning)

    # 5. Split each node into K folds
    superpop_split = Split(SPLIT_DIR, 'ancestry', nodes=nodes)
    if Stage.FOLD_SPLIT in stages:
        logger.info('making k-fold split for the TG dataset')
        splitter = CVSplitter(superpop_split)

        ancestry_df = SplitTGHeter().get_target()
        for node in nodes:
            splitter.split_ids(
                ids_path=os.path.join(SPLIT_GENO_DIR, f'{node}_filtered.psam'),
                node=node,
                y=ancestry_df.set_index('IID')['ancestry'],
                random_state=0
            )

        logger.info(f"Processing split {superpop_split.root_dir}")
        for node in nodes:
            logger.info(f"Saving train, val, test genotypes and running PCA for node {node}")
            for fold_index in range(FOLDS_NUMBER):
                for part_name in ['train', 'val', 'test']:
                    ids_path = superpop_split.get_ids_path(node=node, fold_index=fold_index, part_name=part_name)

                    # Extract and save genotypes
                    run_plink(
                        args_dict={
                            '--pfile': superpop_split.get_source_pfile_path(node=node),
                            '--keep': ids_path,
                            '--out':  superpop_split.get_pfile_path(
                                node=node, fold_index=fold_index, part_name=part_name
                            )
                        },
                        args_list=['--make-pgen']
                    )

                    # write ancestries aka phenotypes
                    relevant_ids = ancestry_df['IID'].isin(pd.read_csv(ids_path, sep='\t')['IID'])
                    ancestry_df.loc[relevant_ids, ['IID', 'ancestry']].to_csv(
                        superpop_split.get_phenotype_path(node=node, fold_index=fold_index, part=part_name),
                        sep='\t', index=False
                    )

        for fold_index in range(FOLDS_NUMBER):
            # Perform centralised sample ids merge to use it with `--keep` flag in plink
            ids = []

            for node in nodes:
                ids_filepath = superpop_split.get_ids_path(
                    fold_index=fold_index,
                    part_name='train',
                    node=node
                )

                ids.extend(pd.read_csv(ids_filepath, sep='\t')['IID'].to_list())

            # Store the list of ids inside the super population split file structure
            centralised_ids_filepath = superpop_split.get_ids_path(
                fold_index=fold_index,
                part_name='train',
                node='ALL'  # centralised PCA
            )

            pd.DataFrame({'IID': ids}).to_csv(centralised_ids_filepath, sep='\t', index=False)

    # 6. PCA
    if Stage.FEDERATED_PCA in stages:
        runner = FederatedPCASimulationRunner(
            source_folder=SPLIT_GENO_DIR,
            result_folder=FEDERATED_PCA_DIR,
            variant_ids_file=os.path.join(SPLIT_GENO_DIR, 'ALL.prune.in'),
            n_components=10,
            method='P-STACK',
            nodes=nodes
        ).run()
    elif Stage.CENTRALIZED_PCA in stages:
        for fold_index in range(FOLDS_NUMBER):
            logger.info(f'Centralised PCA for fold {fold_index}')
            run_plink(
                args_list=[
                    '--pfile', os.path.join(SPLIT_GENO_DIR, 'ALL_filtered'),
                    '--keep', superpop_split.get_ids_path(fold_index=fold_index, part_name='train', node='ALL'),
                    '--freq', 'counts',
                    '--extract', os.path.join(SPLIT_GENO_DIR, 'ALL.prune.in'),
                    '--out', superpop_split.get_pca_path(node='ALL', fold_index=fold_index, part='train', ext=''),
                    '--pca', 'allele-wts', '20'
                ]
            )

            logger.info(f'Projecting train, test, and val parts for each node for fold {fold_index}...')
            for node in nodes:
                for part_name in ['train', 'val', 'test']:
                    run_plink(
                        args_list=[
                            '--pfile', superpop_split.get_pfile_path(
                                node=node, fold_index=fold_index, part_name=part_name
                            ),
                            '--read-freq', superpop_split.get_pca_path(
                                node='ALL', fold_index=fold_index, part='train', ext='.acount'
                            ),
                            '--extract', os.path.join(SPLIT_GENO_DIR, 'ALL.prune.in'),
                            '--score', superpop_split.get_pca_path(
                                node='ALL', fold_index=fold_index, part='train', ext='.eigenvec.allele'
                            ), '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize',
                            '--score-col-nums', '6-25',
                            '--out', superpop_split.get_pca_path(node=node, fold_index=fold_index, part=part_name),
                        ]
                    )
