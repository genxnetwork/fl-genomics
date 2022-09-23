import logging
import os

import pandas
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
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

        '''
        Private method which extracts info about important columns names and positions from header
        '''

        pos_i, pos = self.__search("[Pp][Oo][Ss]*|^[Bb][Pp]$", header)
        id_i, id = self.__search("^[Ii][Dd]*", header)
        p_i, p = self.__search("[Pp].*val*|[Ll][Oo][Gg].*[Pp]*|^[Pp]$", header, pvalue=True)
        chr_i, chr = self.__search("[Cc][Hh][Rr]*", header)
        alt_i, alt = self.__search("[Aa][Ll][Tt]*|^[Aa]1$", header)

        ss_info = {'pos': pos, 'id': id, 'p': p,
                   'pos_i': pos_i, 'id_i': id_i, 'p_i': p_i,
                   'tag': tag, 'chr': chr, 'chr_i': chr_i,
                   'alt': alt, 'alt_i': alt_i}
        cols = [pos, id, chr, alt]
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
                           dtype={ss_info[ss]['chr']: 'string',
                                  }
                           )

        keep_same = {ss_info[ss]['pos'], ss_info[ss]['chr'], ss_info[ss]['alt']}
        df.columns = ['{}{}'.format(c, '' if c in keep_same else '_' + ss_info[ss]['tag']) for c in df.columns]
        new_p = [s + f"_{ss_info[ss]['tag']}" for s in ss_info[ss]['p']]
        for pvalue in new_p:
            df[pvalue] = df[pvalue].astype(float)
            if pvalue == "LOG10_P_UNKNOWN_2":
                continue
            if df[pvalue].max() <= 1:
                logging.info(f'Making log10 {pvalue} in {ss} ...')
                df[pvalue] = np.log10(df[pvalue].replace(0, 0.1**10))
            if df[pvalue].min() < 0:
                logging.info(f'Making -log10 {pvalue} in {ss} ...')
                df[pvalue] = df[pvalue].apply(lambda x: x * -1)
            if norm:
                logging.info(f'Performing min max normalisation for {pvalue}...')
                df[pvalue] = (df[pvalue] - df[pvalue].min()) / \
                             (df[pvalue].max() - df[pvalue].min())

        df_filtered = \
            df.melt(id_vars=[ss_info[ss]['chr'], ss_info[ss]['pos'], ss_info[ss]['alt']], value_vars=new_p)\
                .reset_index(drop=True)
        return df_filtered

    def __merge_data(self, df_dict: dict):

        '''
        Private method which concatenates all summary statistics into one Dataframe
        '''

        df_list = list(df_dict.values())
        for index, _ in enumerate(df_list):
            df_list[index].columns = ['chr', 'pos', 'alt', 'var', 'pvalue']
        return pd.concat(df_list, axis=0, ignore_index=True)

    def __group_data(self, df: pandas.DataFrame, super_pop_list: set):

        '''
        Private method which groups pvalues based on ancestry tags
        '''

        groups = df.groupby('var')
        grouping_dict = {k: [] for k in super_pop_list}
        for name, group in groups:
            for pop in super_pop_list:
                if pop in name:
                    grouping_dict[pop].append(name)

        return groups, grouping_dict

    def prepare_data(self, *ss_and_tag: tuple[str], working_dir: str,
                     to_file='ss.tsv', norm=True) -> pandas.DataFrame:
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
                        merged_df (pandas.DataFrame): Dataframe with columns: chromosome, position, alt allele, pvalue tag, -log10(pvalue)
        '''

        ss_info = {}
        subprocess.call(f"rm {os.path.join(working_dir, 'pos*')}", shell=True)
        for i, ss_tag in enumerate(ss_and_tag):
            ss, tag = ss_tag
            with open(ss, 'r') as ss_file:
                header = ss_file.readline().strip('\n').split('\t')

            logging.info(f'Reading basic info about {ss}...')
            ss_info[ss] = self.__get_ss_info(header, tag)

            logging.info(f'Preparing list of positions for {ss}...')
            pos_list_dist = os.path.join(working_dir, f'pos{i}.list')
            # Prepares a file which contains all snps coords in format <chr>:<position>:<alt_allele>
            subprocess.call("awk -F '\t' '{ print $%s, $%s, $%s }' %s | tail -n +2 | sort | uniq > %s"
                            % (ss_info[ss]['chr_i'], ss_info[ss]['pos_i'], ss_info[ss]['alt_i'], ss, pos_list_dist),
                            shell=True)

        logging.info(f'Making lists of positions intersection...')
        intersection_list_dist = os.path.join(working_dir, f'intersection.list')
        # Makes intersection list of all files with coords prepared earlier
        subprocess.call("""cat %s | sort | uniq -c | awk -F ' ' '{if($1==%s){print $2":"$3":"$4}}' > %s"""
                        % (os.path.join(working_dir, 'pos*'), len(ss_info), intersection_list_dist),
                        shell=True)
        df_dict = {}

        for i, ss_tag in enumerate(ss_and_tag):
            ss, tag = ss_tag
            logging.info(f'Filtering rows based on intersection list for {ss}...')
            # Creates a new column in a summary statistics file containing coords in format <chr>:<position>
            subprocess.call(""" awk '{print $0"\t"$%s":"$%s":"$%s;next}' %s > %s"""
                            % (ss_info[ss]['chr_i'], ss_info[ss]['pos_i'], ss_info[ss]['alt_i'], ss, ss + '.mod'),
                            shell=True)
            # Filters rows of summary statistics file based on intersection
            # list comparing newly created column with coords with intersection list
            subprocess.call("awk 'NR==FNR{a[$1]; next} FNR==1 || $NF in a' %s %s > %s"
                            % (intersection_list_dist, ss + '.mod', ss + '.filtered'),
                            shell=True)

            num_lines_before = sum(1 for _ in open(f'{ss}')) - 1
            num_lines_after = sum(1 for _ in open(f'{ss}.filtered')) - 1
            num_lines_delta = num_lines_before - num_lines_after
            logging.info(f'{num_lines_after} snps left out of {num_lines_before}, '
                         f'total removed: {num_lines_delta} in {ss}')

            logging.info(f'Creating pandas dataframe for {ss}...')
            df = self.__get_dataframe(ss, ss_info, norm)

            df_dict[ss] = df


        logging.info(f'Merging all dataframes into one...')
        df_combined = self.__merge_data(df_dict)
        if to_file:
            logging.info(f'Writing resulting dataframe to {to_file}...')
            df_combined.to_csv(to_file, sep="\t", index=False)
        return df_combined.dropna()


    def plot_scatter(self, df: pandas.DataFrame,
             super_pop_list=('AFR', 'AMR', 'EAS', 'EUR', 'SAS', 'UNKNOWN'),
             top_snps_list=(10, 100, 1000, 2000, 3000, 4000, 5000,
                            6000, 7000, 8000, 9000, 10000),
             to_folder='/home/genxadmin'):

        '''
        Creates simple summary statistics top N snps plot

                Parameters:
                        df (pandas.DataFrame): Dataframe, output of prepare_data
                        super_pop_list (set[str]): List of super population tags, used to group pvalues
                        top_snps_list (set[int]): List of values for x-axis of the plot, amount of top snps for each point on the plot
                        to_folder (str): path to folder to store plots pngs into
        '''

        groups, grouping_dict = self.__group_data(df, super_pop_list)

        for group in grouping_dict:
            if len(grouping_dict[group]) < 2:
                continue
            logging.info(f'Creating scatter plot for {group}...')
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
                    previous = pd.merge(data, previous, how="inner", on=["pos", "chr", "alt"])
                plot_dict[str(value)] = len(previous)

            keys = list(plot_dict.keys())
            vals = [plot_dict[k] for k in keys]
            plt.figure()
            plot = sns.scatterplot(x=keys, y=vals).\
                set(title=f'{group}', xlabel='Number of top snps', ylabel='Number of common snps')
            if to_folder:
                plt.savefig(os.path.join(to_folder, f'{group}_scatter.png'))

    def plot_qq(self, df: pandas.DataFrame,
                percentiles: set[int] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                super_pop_list=('AFR', 'AMR', 'EAS', 'EUR', 'SAS', 'UNKNOWN'),
                to_folder='/home/genxadmin',
                ):

        '''
           Creates a QQ-plot for 2 p-values samples

                   Parameters:
                           df (pandas.DataFrame): Dataframe, output of prepare_data
                           super_pop_list (set[str]): List of super population tags, used to group pvalues (QQ-plot will be created if group has exacly 2 members)
                           percentiles (set[int]): List of percentiles to get values for QQ-plot
                           to_folder (str): path to folder to store plots pngs into
           '''

        groups, grouping_dict = self.__group_data(df, super_pop_list)
        for group in grouping_dict:
            if len(grouping_dict[group]) != 2:
                continue
            logging.info(f'Creating QQ plot for {group}...')
            x = groups.get_group(grouping_dict[group][0])['pvalue']
            y = groups.get_group(grouping_dict[group][1])['pvalue']

            percs = percentiles
            qn_x = np.percentile(x, percs)
            qn_y = np.percentile(y, percs)
            plt.figure()
            fig, ax = plt.subplots()
            ax.set_title(f'{group}')
            ax.set_xlabel(f'Quantiles of {grouping_dict[group][0]}')
            ax.set_ylabel(f'Quantiles of {grouping_dict[group][1]}')
            plt.plot(qn_x, qn_y, ls="", marker="o")

            x = np.linspace(np.min((qn_x.min(), qn_y.min())), np.max((qn_x.max(), qn_y.max())))
            plt.plot(x, x, color="k", ls="--")
            if to_folder:
                plt.savefig(os.path.join(to_folder, f'{group}_qq.png'))

