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

        '''
        Private method which works like regex.search, but returns the matched string or list of strings
        '''

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

        '''
        Private method which extracts info about important columns names and positions from header
        '''

        pos_i, pos = self.__search("[Pp][Oo][Ss]*", header)
        id_i, id = self.__search("^[Ii][Dd]*", header)
        p_i, p = self.__search("[Pp].*val*|[Ll][Oo][Gg].*[Pp]*|^[Pp]$", header, pvalue=True)
        chr_i, chr = self.__search("[Cc][Hh][Rr]*", header)

        ss_info = {'pos': pos, 'id': id, 'p': p,
                   'pos_i': pos_i, 'id_i': id_i, 'p_i': p_i,
                   'tag': tag, 'chr': chr, 'chr_i': chr_i}
        cols = [pos, id, chr]
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

        '''
        Private method which reads filtered summary statistics file and preprocesses it
        '''

        df = pd.read_table(ss + '.filtered', skipinitialspace=True,
                           usecols=ss_info[ss]['cols'],
                           dtype={ss_info[ss]['chr']: 'string'}
                           )
        keep_same = {ss_info[ss]['pos'], ss_info[ss]['chr']}
        df.columns = ['{}{}'.format(c, '' if c in keep_same else '_' + ss_info[ss]['tag']) for c in df.columns]
        new_p = [s + f"_{ss_info[ss]['tag']}" for s in ss_info[ss]['p']]
        for pvalue in new_p:
            if df[pvalue].max() <= 1:
                logging.info(f'Making log10 {pvalue} in {ss} ...')
                df[pvalue] = np.log10(df[pvalue])
            if df[pvalue].min() < 0:
                logging.info(f'Making -log10 {pvalue} in {ss} ...')
                df[pvalue] = df[pvalue].apply(lambda x: x * -1)
            if norm:
                logging.info(f'Performing min max normalisation for {ss}...')
                df[pvalue] = (df[pvalue] - df[pvalue].min()) / \
                             (df[pvalue].max() - df[pvalue].min())

        df_filtered = \
            df.melt(id_vars=[ss_info[ss]['chr'], ss_info[ss]['pos']], value_vars=new_p)\
                .reset_index(drop=True)
        return df_filtered

    def __merge_data(self, df_dict: dict):

        '''
        Private method which concatenates all summary statistics into one Dataframe
        '''

        df_list = list(df_dict.values())
        for index, _ in enumerate(df_list):
            df_list[index].columns = ['chr', 'pos', 'var', 'pvalue']
        return pd.concat(df_list, axis=0, ignore_index=True)

    def prepare_data(self, *ss_and_tag: tuple[str], working_dir: str, to_file='ss.tsv', norm=True) -> pandas.DataFrame:
        '''
        Returns aggregated and intersected summary statistics dataframe.
        All summary statistics files must contain position and chromosome column.
        All necessary columns will be detected automatically with regex.

                Parameters:
                        ss_and_tag (tuple): Any number of tuples with format: (path_to_summary_stat.tsv, ancestry_tag)
                        working_dir (str): Path to a dir to store all lists into
                        to_file (str or None): Path to output dataframe, if None won't create one
                        norm (bool): Perform min max normalisation for each p-value column separately

                Returns:
                        merged_df (pandas.DataFrame): Dataframe with columns: chromosome, position, pvalue tag, -log10(pvalue)
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
            subprocess.call("awk -F '\t' '{ print $%s, $%s }' %s | tail -n +2 | sort | uniq > %s"
                            % (ss_info[ss]['chr_i'], ss_info[ss]['pos_i'], ss, pos_list_dist),
                            shell=True)

        logging.info(f'Making lists of positions intersection...')
        intersection_list_dist = os.path.join(working_dir, f'intersection.list')
        subprocess.call("""cat %s | tail -n +2 | sort | uniq -c | awk -F ' ' '{if($1==%s){print $2":"$3}}' > %s"""
                        % (os.path.join(working_dir, 'pos*'), len(ss_info), intersection_list_dist),
                        shell=True)

        df_dict = {}

        for i, ss_tag in enumerate(ss_and_tag):
            ss, tag = ss_tag
            logging.info(f'Filtering rows based on intersection list for {ss}...')
            subprocess.call(""" awk '{print $0"\t"$1":"$2;next}' %s > %s"""
                            % (ss, ss + '.mod'),
                            shell=True)
            subprocess.call("awk 'NR==FNR{a[$1]; next} FNR==1 || $NF in a' %s %s > %s"
                            % (intersection_list_dist, ss + '.mod', ss + '.filtered'),
                            shell=True)

            logging.info(f'Creating pandas dataframe for {ss}...')
            df = self.__get_dataframe(ss, ss_info, norm)

            df_dict[ss] = df

        logging.info(f'Merging all dataframes into one...')
        df_combined = self.__merge_data(df_dict)
        if to_file:
            logging.info(f'Writing resulting dataframe to {to_file}...')
            df_combined.to_csv(to_file, sep="\t", index=False)
        return df_combined.dropna()


    def plot(self, df: pandas.DataFrame,
             super_pop_list=['AFR', 'AMR', 'EAS', 'EUR', 'SAS'],
             top_snps_list=[10, 20, 30, 40, 50, 100, 1000, 10000],
             to_folder='/home/genxadmin'):

        '''
        Creates simple summary statistics top N snps plot

                Parameters:
                        df (pandas.DataFrame): Dataframe, output of prepare_data
                        super_pop_list (list[str]): List of super population tags, used to group pvalues
                        top_snps_list (list[int]): List of values for x-axis of the plot, amount of top snps for each point on the plot
                        to_folder (str): path to folder to store plots pngs into
        '''

        groups = df.groupby('var')
        grouping_dict = {k: [] for k in super_pop_list}
        for name, group in groups:
            for pop in super_pop_list:
                if pop in name:
                    grouping_dict[pop].append(name)

        for group in grouping_dict:
            if len(grouping_dict[group]) < 2:
                continue
            plot_dict = {}
            for value in top_snps_list:
                g_list = []
                for name in grouping_dict[group]:
                    g = groups.get_group(name)\
                        .sort_values(['pvalue'], ascending=False).head(value)
                    g_list.append(g)

                first_time = True
                for data in g_list:
                    if first_time:
                        previous = data
                        first_time = False
                        continue
                    previous = pd.merge(data, previous, how="inner", on=["pos", "chr"])
                plot_dict[str(value)] = len(previous)

            keys = list(plot_dict.keys())
            vals = [plot_dict[k] for k in keys]
            plt.figure()
            plot = sns.scatterplot(x=keys, y=vals).\
                set(title=f'{group}', xlabel='Number of top snps', ylabel='Number of common snps')
            if to_folder:
                plt.savefig(os.path.join(to_folder, f'{group}.png'))
