import requests
import struct

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

@dataclass
class Layer:
    name: str
    shape: list[int]

    def get_element_number(self):
        total_elements = 1
        for dim in self.shape:
            total_elements *= dim
        return total_elements

class Model:
    def __init__(self):
        self.layers: list[Layer] = []
        
    def reset(self):
        self.layers.clear()

    def add_layer_by_header(self, header: dict):
        for layer_name, layer_info in header.items():
            if layer_name == "__metadata__":
                continue
            self.layers.append(Layer(
                name=layer_name,
                shape=layer_info['shape']
            ))
    
    def get_total_element_number(self):
        total_elements = 0
        for layer in self.layers:
            total_elements += layer.get_element_number()
        return total_elements
    
    def get_layer_number(self):
        return len(self.layers)

    def __repr__(self) -> str:
        return str(self.layers)

def parse_single_file(url):
    '''
    This function implementation is from https://huggingface.co/docs/safetensors/metadata_parsing,
    It can be used to parse the model metadata of a SafeTensors file hosted on the web without downloading the entire file.
    '''
    # Fetch the first 8 bytes of the file
    headers = {'Range': 'bytes=0-7'}
    response = requests.get(url, headers=headers)
    # Interpret the bytes as a little-endian unsigned 64-bit integer
    length_of_header = struct.unpack('<Q', response.content)[0]
    # Fetch length_of_header bytes starting from the 9th byte
    headers = {'Range': f'bytes=8-{7 + length_of_header}'}
    response = requests.get(url, headers=headers)
    # Interpret the response as a JSON object
    header = response.json()
    return header

def create_model_by_url_list(url_list: list[str]):
    model = Model()
    for url in url_list:
        header = parse_single_file(url)
        model.add_layer_by_header(header)
    return model

def merge_model_headers(header1: dict, header2: dict):
    # Merge the two headers
    header1.update(header2)
    return header1
    
def count_elements_in_cpu(model: Model, min_aio_size: int, dtype_size: int):
    in_cpu_elements = 0
    in_cpu_layers = 0
    for layer in model.layers:
        layer_size = layer.get_element_number()
        layer_size_bytes = layer_size * dtype_size
        if layer_size_bytes < min_aio_size:
            in_cpu_elements += layer_size
            in_cpu_layers += 1
    return in_cpu_elements, in_cpu_layers

def main(args: Namespace):
    model = create_model_by_url_list(args.safetensor_url_list)
    in_cpu_elements, in_cpu_layer = count_elements_in_cpu(model, args.min_aio_size, args.dtype_size)
    nvme_elements = model.get_total_element_number() - in_cpu_elements
    nvme_layers = model.get_layer_number() - in_cpu_layer
    print(f"Total elements: {model.get_total_element_number()}")
    print(f"Elements in CPU: {in_cpu_elements}")
    print(f"Elements in NVMe: {nvme_elements}")
    print(f"Percentage in CPU: {in_cpu_elements / model.get_total_element_number() * 100:.2f}%")
    print(f"Total layers: {model.get_layer_number()}")
    print(f"Layers in CPU: {in_cpu_layer}")
    print(f"Layers in NVMe: {nvme_layers}")
    print(f"Percentage in CPU: {in_cpu_layer / model.get_layer_number() * 100:.2f}%")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--safetensor_url_list", type=str, nargs="+", required=True)
    parser.add_argument("--min_aio_size", type=int, required=True)
    parser.add_argument("--dtype_size", type=int, required=True)
    args = parser.parse_args()
    main(args)