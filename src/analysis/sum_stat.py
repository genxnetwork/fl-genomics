import logging
import os

import pandas
import pandas as pd
import numpy as np
import seaborn as sns
import subprocess
import re
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')


class SummaryStat(object):
    '''
    A class to easily aggregate multiple summary statistics files.
    '''

    def __search(self, regex: str, l: list, pvalue=False):
        results_cords = []
        results_names = []
        for i, name in enumerate(l):
            if re.search(regex, name):
                if pvalue:
                    results_cords.append(i + 1)
                    results_names.append(name)
                else:
                    return i + 1, name
        if pvalue:
            return results_cords, results_names
        return False, False

    def __get_ss_info(self, header: list, tag: str):
        pos_i, pos = self.__search("^[Pp][Oo][Ss]*", header)
        id_i, id = self.__search("^[Ii][Dd]*", header)
        p_i, p = self.__search("^[Pp].*val*|^[Ll][Oo][Gg].*[Pp]*|^[Pp]$", header, pvalue=True)

        ss_info = {'pos': pos, 'id': id, 'p': p,
                   'pos_i': pos_i, 'id_i': id_i, 'p_i': p_i,
                   'tag': tag}
        cols = [pos, id]
        for item in cols:
            if not item:
                cols.remove(item)
        for pval in p:
            if not p:
                break
            cols.append(pval)

        ss_info['cols'] = cols
        return ss_info

    def __get_dataframe(self, ss: str, ss_info: dict, norm: bool):
        df = pd.read_table(ss + '.filtered', skipinitialspace=True, usecols=ss_info[ss]['cols'])
        df_filtered = \
            df.drop_duplicates(subset=[ss_info[ss]['pos']]) \
                .sort_values(by=[ss_info[ss]['pos']]) \
                .reset_index(drop=True)

        for pvalue in ss_info[ss]['p']:
            if df_filtered[pvalue].max() > 1:
                df_filtered[pvalue] = np.log10(df_filtered[pvalue])
            if df_filtered[pvalue].min() < 0:
                df_filtered[pvalue] = df_filtered[pvalue].apply(lambda x: x * -1)
            if norm:
                logging.info(f'Performing min max normalisation for {ss}...')
                df_filtered[pvalue] = (df_filtered[pvalue] - df_filtered[pvalue].min()) / \
                             (df_filtered[pvalue].max() - df_filtered[pvalue].min())

        keep_same = {ss_info[ss]['pos'], ss_info[ss]['id']}
        df_filtered.columns = ['{}{}'.format(c, '' if c in keep_same else '_'+ss_info[ss]['tag']) for c in df_filtered.columns]
        return df_filtered

    def __merge_data(self, df_dict: dict, ss_info: dict):
        first_time = True
        for ss in df_dict:
            if first_time:
                previous_ss = df_dict[ss].rename(columns={ss_info[ss]['pos']: 'pos', ss_info[ss]['id']: 'id'})
                first_time = False
                continue
            df = df_dict[ss].rename(columns={ss_info[ss]['pos']: 'pos', ss_info[ss]['id']: 'id'})
            previous_ss = pd.merge(df, previous_ss, how="inner", on=["pos", "id"])

        return previous_ss

    def prepare_data(self, *ss_and_tag: tuple[str], working_dir: str, to_file='ss.tsv', norm=True) -> pandas.DataFrame:
        '''
        Returns aggregated and intersected summary statistics dataframe.
        All summary statistics files must contain position column and at least one of them should have id column.
        All necessary columns will be detected automatically with regex.

                Parameters:
                        ss_and_tag (tuple): Any number of tuples with format: (path_to_summary_stat.tsv, ancestry_tag)
                        working_dir (str): Path to a dir to store all lists into
                        to_file (str or None): Path to output dataframe, if None won't create one
                        norm (bool): Perform min max normalisation for each p-value column separately

                Returns:
                        merged_df (pandas.DataFrame): Dataframe with position, id and all -log10(p-values) columns from all input files
        '''
        ss_info = {}
        for i, ss_tag in enumerate(ss_and_tag):
            ss, tag = ss_tag
            with open(ss, 'r') as ss_file:
                header = ss_file.readline().strip('\n').split('\t')

            logging.info(f'Reading basic info about {ss}...')
            ss_info[ss] = self.__get_ss_info(header, tag)

            logging.info(f'Preparing list of positions for {ss}...')
            pos_list_dist = os.path.join(working_dir, f'pos{i}.list')
            subprocess.call("awk -F '\t' '{ print $%s }' %s | sort | uniq > %s"
                            % (ss_info[ss]['pos_i'], ss, pos_list_dist),
                            shell=True)

        logging.info(f'Making lists of positions intersection...')
        intersection_list_dist = os.path.join(working_dir, f'intersection.list')
        subprocess.call("cat pos* | sort | uniq -c | awk '{if($1==%s){print $2}}' > %s"
                        % (len(ss_info), intersection_list_dist),
                        shell=True)

        df_dict = {}
        found_ids = False
        no_ids_list = []
        for i, ss_tag in enumerate(ss_and_tag):
            ss, tag = ss_tag
            logging.info(f'Filtering rows based on intersection list for {ss}...')
            subprocess.call("awk 'NR==FNR{a[$1]; next} FNR==1 || $%s in a' %s %s > %s"
                            % (ss_info[ss]['pos_i'],intersection_list_dist, ss, ss + '.filtered'),
                            shell=True)

            logging.info(f'Creating pandas dataframe for {ss}...')
            df = self.__get_dataframe(ss, ss_info, norm)

            df_dict[ss] = df

            if ss_info[ss]['id'] and not found_ids:
                id = df[ss_info[ss]['id']]
                found_ids = True
            elif not ss_info[ss]['id']:
                no_ids_list.append(ss)

        if no_ids_list:
            for ss in no_ids_list:
                df_dict[ss]['id'] = id

        logging.info(f'Merging all dataframes into one...')
        df_combined = self.__merge_data(df_dict, ss_info)
        if to_file:
            logging.info(f'Writing resulting dataframe to {to_file}...')
            df_combined.to_csv(to_file, sep="\t", index=False)
        return df_combined

    def plot(self, df: pandas.DataFrame,
             super_pop_list=['AFR', 'AMR', 'EAS', 'EUR', 'SAS'],
             top_snps_list=[10, 20, 30, 40, 50, 100, 1000, 10000],
             to_folder='~'):
        '''
        Returns aggregated and intersected summary statistics dataframe.
        All summary statistics files must contain position column and at least one of them should have id column.
        All necessary columns will be detected automatically with regex.

                Parameters:
                        ss_and_tag (tuple): Any number of tuples with format: (path_to_summary_stat.tsv, ancestry_tag)
                        working_dir (str): Path to a dir to store all lists into
                        to_file (str or None): Path to output dataframe, if None won't create one
                        norm (bool): Perform min max normalisation for each p-value column separately

                Returns:
                        merged_df (pandas.DataFrame): Dataframe with position, id and all -log10(p-values) columns from all input files
        '''

        grouping_dict = {k: [] for k in super_pop_list}
        for col in df.columns:
            for pop in super_pop_list:
                if pop in col:
                    grouping_dict[pop].append(col)

        for group in grouping_dict:
            if len(grouping_dict[group]) < 2:
                continue
            plot_dict = {}
            for value in top_snps_list:
                df_list = []
                for col in grouping_dict[group]:
                    df_g = df[[col, 'id', 'pos']]
                    df_g = df_g.sort_values([col], ascending=False).head(value)
                    df_list.append(df_g)

                first_time = True
                for data in df_list:
                    if first_time:
                        previous = data
                        first_time = False
                        continue
                    previous = pd.merge(data, previous, how="inner", on=["pos", "id"])
                plot_dict[str(value)] = len(previous)

            keys = list(plot_dict.keys())
            vals = [plot_dict[k] for k in keys]
            plt.figure()
            plot = sns.scatterplot(x=keys, y=vals).\
                set(title=f'{group}', xlabel='Number of top snps', ylabel='Number of common snps')
            if to_folder:
                plt.savefig(os.path.join(to_folder, f'{group}.png'))
                
