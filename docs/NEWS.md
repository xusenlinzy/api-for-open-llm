## News or Update

+ 【2023.11.24】 支持 [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 推理，[使用文档](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/LLAMA_CPP.md)


+ 【2023.11.03】 支持 `chatglm3` 和 `qwen` 模型的 `function call` 调用功能，同时支持流式和非流式模式, [工具使用示例](https://github.com/xusenlinzy/api-for-open-llm/tree/master/examples/chatglm3/tool_using.py), 网页 `demo` 已经集成到 [streamlit-demo](./streamlit-demo)


+ 【2023.10.29】 添加 [ChatGLM3](https://github.com/THUDM/ChatGLM3) 模型支持，[启动方式链接](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#chatglm3)，[工具使用示例](https://github.com/xusenlinzy/api-for-open-llm/tree/master/examples/chatglm3)


+ 【2023.09.27】 添加 [Qwen-14B-Chat-Int4](https://huggingface.co/Qwen/Qwen-14B-Chat-Int4) 模型支持，[启动方式链接](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#qwen-14b-chat)


+ 【2023.09.07】 添加 [baichuan2](https://github.com/baichuan-inc/Baichuan2) 模型支持，[启动方式链接](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#baichuan2)


+ 【2023.08.28】 添加 `transformers.TextIteratorStreamer` 流式输出支持，只需将环境变量修改为 `USE_STREAMER_V2=true`


+ 【2023.08.26】 添加 [code-llama](https://github.com/facebookresearch/codellama) 模型支持，[启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#code-llama)，[示例](../examples/code-llama)


+ 【2023.08.23】 添加 [sqlcoder](https://huggingface.co/defog/sqlcoder) 模型支持，[启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#sqlcoder)，[示例](../examples/sqlcoder/inference.py)


+ 【2023.08.22】 添加 [xverse-13b-chat](https://github.com/xverse-ai/XVERSE-13B) 模型支持，[启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#xverse-13b-chat)


+ 【2023.08.10】 添加 [vLLM](https://github.com/vllm-project/vllm) 推理加速支持，[使用方式](./docs/VLLM_SCRIPT.md)


+ 【2023.08.03】 添加 [qwen-7b-chat](https://github.com/QwenLM/Qwen-7B) 模型支持，[启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#qwen-7b-chat)


+ 【2023.07.15】 添加 [starchat](https://huggingface.co/HuggingFaceH4/starchat-beta) 模型支持，[启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#starchat)


+ 【2023.07.12】 添加 [baichuan-13b-chat](https://github.com/baichuan-inc/Baichuan-13B) 模型支持，[启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#baichuan-13b-chat)


+ 【2023.07.07】 添加 [InternLM](https://github.com/InternLM/InternLM) 模型支持，[启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#internlm)


+ 【2023.06.26】 添加 [ChatGLM2-6b](https://github.com/THUDM/ChatGLM2-6B) 模型


+ 【2023.06.12】 使用 [m3e](https://huggingface.co/moka-ai/m3e-base) 中文嵌入模型（在中文文本分类和文本检索上都优于 `openai-ada-002`）
