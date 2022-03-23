import hydra
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from itertools import product
import subprocess
from io import StringIO

def calculate_memory(arg_list):
    node_size_dict = {
        0: 6103,
        1: 1199,
        2: 171578,
        3: 85813,
        4: 42884,
        5: 21444,
        6: 10717,
        7: 3432
    }
    
    for item in args_list:
        if item.split('=')[0].split('.')[-1] == 'node_index':
            node_index = item.split('=')[-1]
        if item.split('=')[0].split('.')[-1] == 'snp_count':
            snp_count = item.split('=')[-1]
    
    return node_size_dict[node_index]*snp_count

def append_args_to_wrapper(arg_list):
    with open('/trinity/home/s.mishra/uk-biobank/src/local/wrapper.sh', 'r') as f:
        wrapper=f.read().rstrip()+" "

    wrapper+=" ".join(arg_list)    
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
        all_args = list(arg_list)+to_add+[f"+model={cfg.model.name}"]
        print(list(arg_list)+to_add)
        wrapper = append_args_to_wrapper(list(arg_list)+to_add)
        print(wrapper)
        process = subprocess.run(['sbatch'],
                                 input=bytes(wrapper, 'utf-8'),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 cwd='/trinity/home/s.mishra/uk-biobank/src')
        
        print(process.stderr)
        print(process.stdout)
if __name__ == '__main__':
    run_grid()
