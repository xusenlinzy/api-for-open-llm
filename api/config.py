import os

import dotenv
from loguru import logger

dotenv.load_dotenv()


DEFAULTS = {
    'HOST': '0.0.0.0',
    'PORT': 8000,

    # support for model
    'MODEL_NAME': '',
    'MODEL_PATH': '',
    'ADAPTER_MODEL_PATH': '',

    # support for device
    'DEVICE': 'cuda',
    'DEVICE_MAP': "",
    'GPUS': '',
    'NUM_GPUs': 1,

    # support for embeddings
    'EMBEDDING_NAME': '',
    'EMBEDDING_SIZE': '',
    'EMBEDDING_DEVICE': 'cuda',

    # support for quantize
    'QUANTIZE': 16,
    'LOAD_IN_8BIT': 'False',
    'LOAD_IN_4BIT': 'False',
    'USING_PTUNING_V2': 'False',

    # support for model input
    'CONTEXT_LEN': '',
    'STREAM_INTERVERL': 2,
    'PROMPT_NAME': '',

    'PATCH_TYPE': '',
    'ALPHA': 'auto',

    'API_PREFIX': '/v1',

    # support for vllm
    'USE_VLLM': 'False',
    'TRUST_REMOTE_CODE': "False",
    'TOKENIZE_MODE': "auto",
    'TENSOR_PARALLEL_SIZE': 1,
    'DTYPE': "half",
    "GPU_MEMORY_UTILIZATION": 0.9,
    "MAX_NUM_BATCHED_TOKENS": 5120,
    "MAX_NUM_SEQS": 256,

    # support for transformers.TextIteratorStreamer
    'USE_STREAMER_V2': 'False',

    # support for api key check
    'API_KEYS': '',

    'ACTIVATE_INFERENCE': 'True',
}


def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))


def get_bool_env(key):
    return get_env(key).lower() == 'true'


class Config:
    """ Configuration class. """

    def __init__(self):
        self.HOST = get_env('HOST')
        self.PORT = int(get_env('PORT'))

        self.MODEL_NAME = get_env('MODEL_NAME')
        self.MODEL_PATH = get_env('MODEL_PATH')
        self.ADAPTER_MODEL_PATH = get_env('ADAPTER_MODEL_PATH') if get_env('ADAPTER_MODEL_PATH') else None

        self.DEVICE = get_env('DEVICE')
        self.DEVICE_MAP = get_env('DEVICE_MAP') if get_env('DEVICE_MAP') else None
        self.GPUS = get_env('GPUS')
        self.NUM_GPUs = int(get_env('NUM_GPUs'))

        self.EMBEDDING_NAME = get_env('EMBEDDING_NAME') if get_env('EMBEDDING_NAME') else None
        self.EMBEDDING_SIZE = int(get_env('EMBEDDING_SIZE')) if get_env('EMBEDDING_SIZE') else None
        self.EMBEDDING_DEVICE = get_env('EMBEDDING_DEVICE')

        self.QUANTIZE = int(get_env('QUANTIZE'))
        self.LOAD_IN_8BIT = get_bool_env('LOAD_IN_8BIT')
        self.LOAD_IN_4BIT = get_bool_env('LOAD_IN_4BIT')
        self.USING_PTUNING_V2 = get_bool_env('USING_PTUNING_V2')

        self.CONTEXT_LEN = int(get_env('CONTEXT_LEN')) if get_env('CONTEXT_LEN') else None
        self.STREAM_INTERVERL = int(get_env('STREAM_INTERVERL'))
        self.PROMPT_NAME = get_env('PROMPT_NAME') if get_env('PROMPT_NAME') else None

        self.PATCH_TYPE = get_env('PATCH_TYPE') if get_env('PATCH_TYPE') else None
        self.ALPHA = get_env('ALPHA')

        self.API_PREFIX = get_env('API_PREFIX')

        self.USE_VLLM = get_bool_env('USE_VLLM')
        self.TRUST_REMOTE_CODE = get_bool_env('TRUST_REMOTE_CODE')
        self.TOKENIZE_MODE = get_env('TOKENIZE_MODE')
        self.TENSOR_PARALLEL_SIZE = int(get_env('TENSOR_PARALLEL_SIZE'))
        self.DTYPE = get_env('DTYPE')
        self.GPU_MEMORY_UTILIZATION = float(get_env('GPU_MEMORY_UTILIZATION'))
        self.MAX_NUM_BATCHED_TOKENS = int(get_env('MAX_NUM_BATCHED_TOKENS'))
        self.MAX_NUM_SEQS = int(get_env('MAX_NUM_SEQS'))

        self.USE_STREAMER_V2 = get_bool_env('USE_STREAMER_V2')

        self.API_KEYS = get_env('API_KEYS').split(',') if get_env('API_KEYS') else None

        self.ACTIVATE_INFERENCE = get_bool_env('ACTIVATE_INFERENCE')


config = Config()
logger.debug(f"Config: {config.__dict__}")
if config.GPUS:
    if len(config.GPUS.split(",")) < config.NUM_GPUs:
        raise ValueError(
            f"Larger --num_gpus ({config.NUM_GPUs}) than --gpus {config.GPUS}!"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
