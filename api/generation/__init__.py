from api.generation.baichuan import (
    build_baichuan_chat_input,
    check_is_baichuan,
)
from api.generation.chatglm import (
    generate_stream_chatglm,
    check_is_chatglm,
    generate_stream_chatglm_v3,
)
from api.generation.qwen import (
    build_qwen_chat_input,
    check_is_qwen,
)
from api.generation.stream import (
    generate_stream,
    generate_stream_v2,
)
from api.generation.xverse import (
    build_xverse_chat_input,
    check_is_xverse,
)
