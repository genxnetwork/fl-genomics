import socket
from typing import List
import numpy


def is_port_available(port: int):
    result = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
            result = True
        except:
            print(f'Port {port} is in use')
    return result

def get_available_ports(nodes: int = 2) -> List[int]:
    
    ports = numpy.arange(47000, 48000, dtype=int)
    numpy.random.shuffle(ports)
    chosen_ports = []
    port_index = 0
    for node in range(nodes):
        for pi in range(port_index, ports.shape[0]):
            if is_port_available(ports[pi]):
                chosen_ports.append(ports[pi])
                port_index = pi + 1
                break
            
    if len(chosen_ports) == nodes:
        return chosen_ports
        
    print(f'We failed after attempting to find an available port in {ports.shape[0]} ports for node {node}')
    raise RuntimeError(f'There are no available port in {ports.shape[0]} for {nodes}')