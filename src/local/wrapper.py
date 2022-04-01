import hydra
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from itertools import product
import subprocess
from io import StringIO
import os

def calculate_memory(node_index, split, snp_count):
    node_size_dict = {
    'uneven_split': {
        0: 6103,
        1: 1199,
        2: 171578,
        3: 85813,
        4: 42884,
        5: 21444,
        6: 10717,
        7: 3432
    },
    'ethnic_split': {
        0: 343858,
        1: 6139,
        2: 6086,
        3: 1201,
        4: 30317 
    }}
    
    mem = int(5000+node_size_dict[split][node_index]*snp_count/32000)
    return mem

def append_args_to_wrapper(arg_list):
    with open(f"/trinity/home/{os.environ['USER']}/uk-biobank/src/local/wrapper.sh", 'r') as f:
        wrapper=f.read().rstrip()+" "

    run_experiment = None 
    
    for arg in arg_list:
        if arg.split('=')[0] == 'model.name':
            model_name = arg.split('=')[1]
        if arg.split('=')[0] == 'node_index':
            node_index = int(arg.split('=')[1])
        if arg.split('=')[0] == 'experiment.snp_count':
            snp_count = int(arg.split('=')[1])
        if arg.split('=')[0] == 'experiment.run':
            run_experiment = arg.split('=')[1]
        if arg.split('=')[0] == 'split_dir':
            split = arg.split('=')[1].split('/')[-1]

    mem = calculate_memory(node_index, split, snp_count)
    wrapper = wrapper.replace('$MEM', str(mem))
    wrapper += f"+model={model_name} "

    if run_experiment:
        arg_list.remove(f'experiment.run={run_experiment}')
        wrapper += f"+experiment={run_experiment} "
    
    wrapper += " ".join(arg_list) 
    return wrapper

@hydra.main(config_path='configs/grid', config_name='default')
def run_grid(cfg: DictConfig):
    to_multiply = []
    to_add = []
    
    def traverse(d, s=''):
        for k,v in d.items():
            print(type(v))
            if isinstance(v, DictConfig):
                traverse(v, s+k+'.')
            elif isinstance(v, ListConfig):
                to_multiply.append([s+k+'='+str(value) for value in v])
            else:
                to_add.append(s+k+'='+str(v))
        
    traverse(cfg)
    
    for arg_list in list(product(*to_multiply)):
        print(list(arg_list)+to_add)
        wrapper = append_args_to_wrapper(list(arg_list)+to_add)
        print(wrapper)
        process = subprocess.run(['sbatch'],
                                 input=bytes(wrapper, 'utf-8'),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 cwd=f"/trinity/home/{os.environ['USER']}/uk-biobank/src")
        
        print(process.stderr)
        print(process.stdout)

if __name__ == '__main__':
    run_grid()
