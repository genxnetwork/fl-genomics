from typing import Dict, List
import matplotlib.pyplot as plt
import os


def plot_snp_sample_dependence(x: List[int], data: Dict):
    plt.figure(figsize=(15, 10))
    for snp_count, y in data.items():
        plt.plot(x, y, marker='o', markersize=16, linewidth=2, label=f'{snp_count} SNPs')
    
    plt.ylabel('Test $R^2$', fontsize=20)
    plt.xlabel('Training Samples', fontsize=20)
    plt.title(f'MLP Test $R^2$ for British Split Individuals', fontsize=24)
    plt.legend(fontsize=20)
    
    plt.grid()
    
    xticks, xticklabels = plt.xticks()
    print(xticks, xticklabels)
    plt.xticks(xticks, [str(int(tick) // 1000) + 'K' for tick in xticks], fontsize=18)

    yticks, yticklabels = plt.yticks()
    plt.yticks(yticks, [f'{tick:.2f}' for tick in yticks], fontsize=18)
    plt.xlim((40000, 180000))
    print(f'saving figure to {os.getcwd()}/snp_sample_dependence.png')
    plt.savefig('snp_sample_dependence.png')
    


if __name__ == '__main__':
    print(f'plotting started')
    x = [171578, 85813, 42884]
    data = {
        '2000' : [0.6, 0.592, 0.578],
        '5000': [0.63, 0.61, 0.591],
        '10000': [0.653, 0.629, 0.591]
    }
    plot_snp_sample_dependence(x, data)