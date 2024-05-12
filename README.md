# Safetensors-Parser

This is a parser that you can provide model safetensors web host file and it will parse the file and return the model information.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Modify the `run.sh` file.
1. Provide the `url` to the model file, you can provide 1 or more files url.
2. Provide `min_aio_size`, this size is represented in bytes.
    - In deepspeed zero infinity implementation, when the layer size is less than this size, it will be stay to CPU, otherwise it will be moved to NVMe.
    - This value is calculated by `max(1MB, AIO_BLOCK_SIZE)` where `AIO_BLOCK_SIZE` is the chunk size of the deepspeed aio library. Where deepspeed aio library will read or write the data in chunks.
3. provide `dtype_size`, this size is represented for the model data type.
    - For example, during forward and backward pass, the model will be stored in fp16, then the value will be 2.
    - And during the optimizer step, the model may be stored in fp32, then the value will be 4.

### Run the script
```bash
bash run.sh
```

### Output

The output will in this format:
```plaintext
Total elements: 8030261248
Elements in CPU: 266240
Elements in NVMe: 8029995008
Percentage in CPU: 0.00%
Total layers: 291
Layers in CPU: 65
Layers in NVMe: 226
Percentage in CPU: 22.34%
```
1. Total elements: Total number of elements in the model.
2. Elements in CPU: Total number of elements that will be stored in CPU.
3. Elements in NVMe: Total number of elements that will be stored in NVMe.
4. Percentage in CPU: Percentage of elements that will be stored in CPU.
5. Total layers: Total number of layers in the model.
6. Layers in CPU: Total number of layers that will be stored in CPU.
7. Layers in NVMe: Total number of layers that will be stored in NVMe.
8. Percentage in CPU: Percentage of layers that will be stored in CPU.