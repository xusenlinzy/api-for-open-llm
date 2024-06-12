import api.templates.registry
from api.templates.baichuan import BaiChuanChatTemplate, BaiChuan2ChatTemplate
from api.templates.base import ChatTemplate
from api.templates.glm import (
    ChatGLMChatTemplate,
    ChatGLM2ChatTemplate,
    ChatGLM3ChatTemplate,
    ChatGLM4ChatTemplate,
    GLM4VChatTemplate,
)
from api.templates.qwen import QwenChatTemplate, Qwen2ChatTemplate
from api.templates.registry import register_template, get_template

__all__ = [
    "BaiChuanChatTemplate",
    "BaiChuan2ChatTemplate",
    "ChatGLMChatTemplate",
    "ChatGLM2ChatTemplate",
    "ChatGLM3ChatTemplate",
    "ChatGLM4ChatTemplate",
    "GLM4VChatTemplate",
    "QwenChatTemplate",
    "Qwen2ChatTemplate",
    "Llama2ChatTemplate",
    "Llama3ChatTemplate",
    "ChineseAlpaca2ChatTemplate",
    "AquilaChatTemplate",
    "FireflyChatTemplate",
    "FireflyQwenChatTemplate",
    "BelleChatTemplate",
    "OpenBuddyChatTemplate",
    "InternLMChatTemplate",
    "InternLM2ChatTemplate",
    "SusChatTemplate",
    "StarChatTemplate",
    "OrionStarChatTemplate",
    "DeepseekChatTemplate",
    "HuatuoChatTemplate",
    "MistralChatTemplate",
    "VicunaChatTemplate",
    "XuanYuanChatTemplate",
    "BlueLMChatTemplate",
    "DeepseekCoderChatTemplate",
    "YiAIChatTemplate",
    "ChatMLTemplate",
]


@register_template("llama2")
class Llama2ChatTemplate(ChatTemplate):
    system_prompt = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    )
    stop = ["[INST]", "[/INST]"]


@register_template("llama3")
class Llama3ChatTemplate(ChatTemplate):
    system_prompt = ""
    stop = ["<|end_of_text|>", "<|eot_id|>"]
    stop_token_ids = [128001, 128009]


@register_template("chinese-llama-alpaca2")
class ChineseAlpaca2ChatTemplate(Llama2ChatTemplate):
    system_prompt = "You are a helpful assistant. 你是一个乐于助人的助手。"


@register_template("alpaca")
class AlpacaChatTemplate(ChatTemplate):
    system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    stop = ["### Instruction", "### Response"]

    @property
    def chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}"
            "{% else %}"
            "{{ system_prompt }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '### Instruction:\\n' + message['content'] + '\\n\\n### Response:\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '\\n\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("firefly")
class FireflyChatTemplate(ChatTemplate):
    system_prompt = "<s>"

    @property
    def chat_template(self) -> str:
        return (
            "{{ system_prompt }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] + '</s>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '</s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("firefly-qwen")
class FireflyQwenChatTemplate(ChatTemplate):
    system_prompt = "<|endoftext|>"

    @property
    def chat_template(self) -> str:
        return (
            "{{ system_prompt }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] + '<|endoftext|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '<|endoftext|>' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("belle")
class BelleChatTemplate(ChatTemplate):

    @property
    def chat_template(self) -> str:
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ 'Human: ' + message['content'] + '\\n\\nAssistant: ' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '\\n\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("openbuddy")
class OpenBuddyChatTemplate(ChatTemplate):
    system_prompt = """Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team, based on Falcon and LLaMA Transformers architecture. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, and more.
Buddy possesses knowledge about the world, history, and culture, but not everything. Knowledge cutoff: 2021-09.
Buddy's responses are always positive, unharmful, safe, creative, high-quality, human-like, and interesting.
Buddy must always be safe and unharmful to humans.
Buddy strictly refuses to discuss harmful, political, NSFW, illegal, abusive, offensive, or other sensitive topics.
"""

    @property
    def chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}"
            "{% else %}"
            "{{ system_prompt + '\\n' }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ 'User: ' + message['content'] + '\\nAssistant: ' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '\\n\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("internlm")
class InternLMChatTemplate(ChatTemplate):
    stop = ["</s>", "<eoa>"]

    @property
    def chat_template(self) -> str:
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<s><|User|>:' + message['content'] + '<eoh>\\n<|Bot|>:' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '<eoa>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("internlm2")
class InternLM2ChatTemplate(ChatTemplate):
    system_prompt = (
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
    )
    stop = ["</s>", "<|im_end|>"]

    @property
    def chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ '<s><|im_start|>' + 'system\\n' + messages[0]['content'] + '<|im_end|>' + '\\n' }}"
            "{% else %}"
            "{{ '<s><|im_start|>' + 'system\\n' + system_prompt + '<|im_end|>' + '\\n' }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )


@register_template("starchat")
class StarChatTemplate(ChatTemplate):
    stop = ["<|end|>"]
    stop_token_ids = [49152, 49153, 49154, 49155]

    @property
    def chat_template(self) -> str:
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>\\n' }}"
            "{% endif %}"
        )


@register_template("aquila")
class AquilaChatTemplate(ChatTemplate):
    stop = ["###", "[UNK]", "</s>"]

    @property
    def chat_template(self) -> str:
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ 'Human: ' + message['content'] + '###' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ 'System: ' + message['content'] + '###' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ 'Assistant: ' + message['content'] + '###' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ 'Assistant: ' }}"
            "{% endif %}"
        )


@register_template("vicuna")
class VicunaChatTemplate(ChatTemplate):
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

    @property
    def chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}"
            "{% else %}"
            "{{ system_prompt }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ 'USER: ' + message['content'] + ' ASSISTANT: ' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '</s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("xuanyuan")
class XuanYuanChatTemplate(ChatTemplate):
    system_prompt = "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n"

    @property
    def chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}"
            "{% else %}"
            "{{ system_prompt }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ 'Human: ' + message['content'] + 'Assistant: ' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '</s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("deepseek-coder")
class DeepseekCoderChatTemplate(ChatTemplate):
    stop = ["<|EOT|>"]


@register_template("deepseek")
class DeepseekChatTemplate(ChatTemplate):
    stop = ["<｜end▁of▁sentence｜>"]
    stop_token_ids = [100001]


@register_template("bluelm")
class BlueLMChatTemplate(ChatTemplate):
    stop = ["[|Human|]", "[|AI|]"]

    @property
    def chat_template(self) -> str:
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '[|Human|]:' + message['content'] + '[|AI|]:' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '</s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("huatuo")
class HuatuoChatTemplate(ChatTemplate):
    system_prompt = "一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问诊，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。"
    stop_token_ids = [195, 196]
    stop = ["<reserved_102>", "<reserved_103>", "<病人>"]

    @property
    def chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}"
            "{% else %}"
            "{{ system_prompt }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<病人>：' + message['content'] + ' <HuatuoGPT>：' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '</s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )


@register_template("orionstar")
class OrionStarChatTemplate(ChatTemplate):
    """ https://huggingface.co/OrionStarAI/Orion-14B-Chat/blob/4de9f928abf60f8f3a3f4d7f972f4807aa57c573/generation_utils.py#L12 """
    stop = ["</s>"]


@register_template("yi")
class YiAIChatTemplate(ChatTemplate):
    """ https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/tokenizer_config.json """
    stop = ["<|endoftext|>", "<|im_end|>"]
    stop_token_ids = [2, 6, 7, 8]  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"


@register_template("suschat")
class SusChatTemplate(ChatTemplate):
    """ https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/tokenizer_config.json """
    stop_token_ids = [2]
    stop = ["<|endoftext|>", "### Human"]


@register_template("mistral")
class MistralChatTemplate(ChatTemplate):
    stop = ["[INST]", "[/INST]"]


@register_template("chatml")
class ChatMLTemplate(ChatTemplate):
    stop = ["<|endoftext|>", "<|im_end|>"]
