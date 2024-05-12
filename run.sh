LLAMA3_8B_SAFETENSOR_URL_LIST="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k/resolve/main/model-00001-of-00004.safetensors https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k/resolve/main/model-00002-of-00004.safetensors https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k/resolve/main/model-00003-of-00004.safetensors https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k/resolve/main/model-00004-of-00004.safetensors"
BLOOMZ_560M_SAFETENSOR_URL_LIST="https://huggingface.co/bigscience/bloomz-560m/resolve/main/model.safetensors"
BLOOMZ_3B_SAFETENSOR_URL_LIST="https://huggingface.co/bigscience/bloomz-3b/resolve/main/model.safetensors"

SAFETENSOR_URL_LIST=$LLAMA3_8B_SAFETENSOR_URL_LIST

MIN_AIO_SIZE=2097152    # 2MB
DTYPE_SIZE=2    # fp16

python -u main.py \
    --safetensor_url_list $SAFETENSOR_URL_LIST \
    --min_aio_size $MIN_AIO_SIZE \
    --dtype_size $DTYPE_SIZE
