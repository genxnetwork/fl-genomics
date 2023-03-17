# partially copied from src/JupyterNotebooks/tg_ancestries.ipynb
import logging
import os.path

import pandas as pd
import pickle
import numpy as np
import torch

import plotly.express as px
import matplotlib.pyplot as plt

import sys

from sklearn.metrics import confusion_matrix

sys.path.append('..')

from configs.split_config import TG_SUPERPOP_DICT, ethnic_background_name_map

# Arguments for projecting into TG PC space with plink

# plink2 --extract variants --out ukb_1kg_projections --pfile /gpfs/gpfs0/ukb_data/plink/plink --read-freq tg_pca.acount --score tg_pca.eigenvec.allele 2 5 variance-standardize --score-col-nums 6-25

DATA_FOLDER = '/home/dkolobok/data/fl-genomics'
MODELS_FOLDER = os.path.join(DATA_FOLDER, 'models')
OUT_FOLDER = os.path.join(DATA_FOLDER, 'out')
UKB_TG_PROJECTIONS_PATH = os.path.join(DATA_FOLDER, 'ukb_tg_projections_100.sscore')


SUPERPOPULATIONS_OUTPUT_PATH = os.path.join(DATA_FOLDER, 'superpopulations.csv')

SUPERPOP_NODE_MAP = {
    'EUR': 0,
    'SAS': 1,
    'AFR': 2,
    'EAS': 3,
    'AMR': 4
}


class UkbAncestryTg(object):
    def __init__(self, num_pcs=20):
        logger.info(f"Processing {num_pcs} PCs")
        self.num_pcs = num_pcs
        self.tg_tag = f'tg_pca_{num_pcs}pcs'
        self.tg_pca_path = os.path.join(MODELS_FOLDER, self.tg_tag + '.sscore')
        self.tg_ancestry_model_path = os.path.join(MODELS_FOLDER, self.tg_tag + '.pkl')
        self.populations = list(sorted(TG_SUPERPOP_DICT.keys()))

    def first(self):
        df_tg = pd.read_table(self.tg_pca_path)
        df_tg = df_tg.drop(['ALLELE_CT', 'NAMED_ALLELE_DOSAGE_SUM'], axis=1)
        df_tg['pop'] = 'tg'
        # df_ukb = pd.read_table(UKB_TG_PROJECTIONS_PATH)
        # df_ukb = df_ukb.drop(['#FID', 'ALLELE_CT', 'NAMED_ALLELE_DOSAGE_SUM'], axis=1)[::10]
        # df_ukb['pop'] = 'ukb'
        # df_ukb.columns = df_tg.columns
        # df = pd.concat([df_ukb, df_tg])
        # px.scatter(df, x='PC1_AVG', y='PC2_AVG', color='pop').write_html('/home/dkolobok/tmp.html')
        return df_tg

    def predictions_tg(self, ancestry_model, df_tg):
        X = df_tg.filter(like="PC").values
        pred_probs = torch.softmax(ancestry_model.forward(torch.Tensor(X)), dim=1).detach().numpy()

        df_tg['pred_ancestry'] = np.vectorize(lambda index: self.populations[index])(np.argmax(pred_probs, axis=1))
        df_tg['pred_superpop'] = df_tg.pred_ancestry.map(TG_SUPERPOP_DICT)
        df_tg['node_index'] = df_tg.pred_superpop.map(SUPERPOP_NODE_MAP)
        tg_pops = pd.read_csv(os.path.join(DATA_FOLDER, 'tg_pca.tsv'), sep='\t').rename(columns={'IID': '#IID'})
        df_tg = pd.merge(df_tg, tg_pops, on='#IID', how='outer')
        self.acc_tg = (df_tg['pred_ancestry'] == df_tg['ancestry']).sum() / len(df_tg)
        logger.info(f"Model accuracy on TG: {self.acc_tg}")
        df_gbr = df_tg.query("ancestry == 'GBR'")
        self.acc_tg_gbr = (df_gbr['pred_ancestry'] == df_gbr['ancestry']).sum() / len(df_gbr)
        logger.info(f"Model accuracy on GBR samples from TG: {self.acc_tg_gbr}")
        cm = pd.crosstab(df_tg['ancestry'], df_tg['pred_ancestry'])
        px.imshow(cm, x=cm.columns, y=cm.index).write_html(os.path.join(OUT_FOLDER, f'tg_conf_matrix_{self.num_pcs}pcs.html'))
        pass

    def predictions_ukb(self, ancestry_model, ukb_tg_projections):
        X = ukb_tg_projections.filter(like='AVG').values
        pred_probs = torch.softmax(ancestry_model.forward(torch.Tensor(X)), dim=1).detach().numpy()

        ukb_tg_projections['pred_ancestry'] = np.vectorize(lambda index: self.populations[index])(np.argmax(pred_probs, axis=1))
        ukb_tg_projections['pred_superpop'] = ukb_tg_projections.pred_ancestry.map(TG_SUPERPOP_DICT)
        ukb_tg_projections['node_index'] = ukb_tg_projections.pred_superpop.map(SUPERPOP_NODE_MAP)

        superpop_list = np.vectorize(lambda x: SUPERPOP_NODE_MAP[TG_SUPERPOP_DICT[x]])(self.populations)
        superpop_probs = np.zeros((ukb_tg_projections.shape[0], len(SUPERPOP_NODE_MAP)))
        for i in range(len(SUPERPOP_NODE_MAP)):
            superpop_probs[:, i] = np.sum(pred_probs[:, superpop_list == i], axis=1)

        ukb_tg_projections['superpop_confidence'] = np.max(superpop_probs, axis=1)

        ukb_tg_projections = self.add_ukb_reported_ancestry(ukb_tg_projections)

        ukb_tg_projections.loc[ukb_tg_projections.superpop_confidence > 0.95, ['IID', 'node_index']].to_csv(SUPERPOPULATIONS_OUTPUT_PATH, index=False)
        # logger.info(f"Predicted superpopulations for UKB samples: \n{ukb_tg_projections.pred_superpop.value_counts()}")
        ukb_tg_projections_brit = ukb_tg_projections.query("ethnic_background == 'British'")
        self.acc_ukb_brit_pred_gbr = (ukb_tg_projections_brit['pred_ancestry'] == 'GBR').sum() / len(ukb_tg_projections_brit)
        self.acc_ukb_brit_pred_ceu = (ukb_tg_projections_brit['pred_ancestry'] == 'CEU').sum() / len(ukb_tg_projections_brit)
        logger.info(f"Predicted GBR and CEU populations for British UKB samples: \n{ukb_tg_projections_brit.pred_ancestry.value_counts().loc[['GBR', 'CEU']]}")
        return ukb_tg_projections

    def second(self, df_tg):
        ukb_tg_projections = pd.read_table(UKB_TG_PROJECTIONS_PATH)[['IID'] + [f'SCORE{i}_AVG' for i in range(1, self.num_pcs + 1)]]
        ancestry_model = pickle.load(open(self.tg_ancestry_model_path, 'rb'))
        # with open(os.path.join(MODELS_FOLDER, 'centr.pkl'), "rb") as input_file:
        #     ancestry_model = pickle.load(input_file)
        ancestry_model.eval()
        self.predictions_tg(ancestry_model=ancestry_model, df_tg=df_tg)
        ukb_tg_projections = self.predictions_ukb(ancestry_model=ancestry_model, ukb_tg_projections=ukb_tg_projections)
        # px.scatter(ukb_tg_projections, x='SCORE1_AVG', y='SCORE2_AVG', color='pred_ancestry').write_html('ancestries_pc1v2.html')
        # px.scatter(ukb_tg_projections, x='SCORE3_AVG', y='SCORE4_AVG', color='pred_ancestry').write_html('ancestries_pc3v4.html')
        # px.scatter(ukb_tg_projections, x='SCORE3_AVG', y='SCORE4_AVG', color='pred_superpop').write_html('superpopulations_pc3v4.html')
        # px.scatter(ukb_tg_projections, x='SCORE1_AVG', y='SCORE2_AVG', color='pred_superpop').write_html('superpopulations_pc1v2.html')
        return ukb_tg_projections

    def add_ukb_reported_ancestry(self, ukb_tg_projections):
        df = pd.read_csv(os.path.join(DATA_FOLDER, 'ethnic_backgrounds.tsv'), sep='\t')
        df['ethnic_background'] = df['ethnic_background'].replace(ethnic_background_name_map)
        return pd.merge(ukb_tg_projections, df[['IID', 'ethnic_background']], on='IID', how='left')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    logger = logging.getLogger()
    stats = pd.DataFrame(columns=['acc_tg', 'acc_tg_gbr', 'acc_ukb_brit_pred_gbr', 'acc_ukb_brit_pred_ceu'])
    for num_pcs in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        uat = UkbAncestryTg(num_pcs=num_pcs)
        df_tg = uat.first()
        ukb_tg_projections = uat.second(df_tg=df_tg)
        stats.loc[num_pcs, :] = [uat.acc_tg, uat.acc_tg_gbr, uat.acc_ukb_brit_pred_gbr, uat.acc_ukb_brit_pred_ceu]
        # uat.third(ukb_tg_projections=ukb_tg_projections)
    pass