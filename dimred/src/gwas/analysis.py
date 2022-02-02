import os
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import pandas
import numpy
from qmplot import qqplot


def plot_manhattan(gwas: pandas.DataFrame, output_path: str):
    to_plot = gwas.copy(deep=True)
    to_plot.sort_values(by=['#CHROM', 'POS'], inplace=True)
    to_plot.loc[:, 'ind'] = range(len(to_plot))
    data_grouped = to_plot.groupby(('#CHROM'))

    fig = plt.figure(figsize=(17, 12))
    ax = fig.add_subplot(111)
    ax.grid(which='both')
    colors = ['red','green','blue']
    x_labels = []
    x_labels_pos = []

    for num, (name, group) in enumerate(data_grouped):
        group.plot(kind='scatter', x='ind', y='LOG10_P', 
                   s=16, linewidths=0.5, edgecolors='gray', color=colors[num % len(colors)], ax=ax)
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
        
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([0, len(to_plot)])
    ax.set_ylim([0, 20])
    ax.set_xlabel('Chromosome')

    fig.savefig(output_path)

    
def plot_hist(gwas: pandas.DataFrame, output_path: str):
    x = gwas['LOG10_P'].values
    plt.grid(which='both')
    plt.figure(figsize=(15, 10))
    plt.hist(x, bins=50, )
    plt.savefig(output_path)


def plot_qq(gwas: pandas.DataFrame, output_path: str):
    x = gwas['LOG10_P'].values
    qqplot(10**(-x[x<=20]), figname=output_path)


def plot_log10p(gwas: pandas.DataFrame, output_path: str):
    # plink 2.0 actually outputs -log10(p), therefore greater ['LOG10_P'] values correspond to smaller p-values
    y = numpy.sort(gwas['LOG10_P'].values)[::-1] # we will place greater values first

    plt.figure(figsize=(15, 10))
    plt.grid(True, which='both')
    plt.xlabel('SNP number')
    plt.ylabel('-LOG10(P)')
    plt.title('Sorted -LOG10(P) values')
    plt.plot(y, linewidth=2, label='-LOG10(P)')
    plt.savefig(output_path)


@hydra.main(config_path='configs', config_name='gwas')
def main(cfg: DictConfig):

    gwas_results_path = f'{cfg.output.path}.{cfg.phenotype.name}.tsv'
    # pvalues_plot_path = f'{cfg.output.path}.log10p.png'
    qq_plot_path = f'{cfg.output.path}.qqplot.png'
    hist_plot_path = f'{cfg.output.path}.hist.png'
    manhattan_plot_path = f'{cfg.output.path}.manhattan.png'

    gwas = pandas.read_table(gwas_results_path)
    # plot_log10p(gwas, pvalues_plot_path)
    plot_qq(gwas, qq_plot_path)
    plot_hist(gwas, hist_plot_path)
    plot_manhattan(gwas, manhattan_plot_path)

    print(f'GWAS analysis for phenotype {cfg.phenotype.name} and split_index {cfg.split_index} finished')


if __name__ == '__main__':
    main()