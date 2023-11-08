import os

import streamlit as st
from openai import OpenAI

from .utils import CodeKernel, extract_code, execute, postprocess_text


@st.cache_resource
def get_kernel():
    return CodeKernel()


SYSTEM_MESSAGE = [
    {
        "role": "system",
        "content": "你是一位智能AI助手，你叫ChatGLM，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。"
    }
]


def chat_once(message_placeholder, client: OpenAI):
    params = dict(
        model="chatglm3",
        messages=SYSTEM_MESSAGE + st.session_state.messages,
        stream=True,
        max_tokens=st.session_state.get("max_tokens", 512),
        temperature=st.session_state.get("temperature", 0.9),
    )
    response = client.chat.completions.create(**params)

    display = ""
    for _ in range(5):
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            display += content
            message_placeholder.markdown(postprocess_text(display) + "▌")

            if chunk.choices[0].finish_reason == "stop":
                message_placeholder.markdown(postprocess_text(display) + "▌")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response
                    }
                )
                return

            elif chunk.choices[0].finish_reason == "function_call":
                try:
                    code = extract_code(full_response)
                except:
                    continue

                with message_placeholder:
                    with st.spinner("Executing code..."):
                        try:
                            res_type, res = execute(code, get_kernel())
                        except Exception as e:
                            st.error(f"Error when executing code: {e}")
                            return

                if res_type == "text":
                    res = postprocess_text(res)
                    display += "\n" + res
                    message_placeholder.markdown(postprocess_text(display) + "▌")
                elif res_type == "image":
                    st.image(res)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "function_call": {"name": "interpreter", "arguments": ""},
                    }
                )
                st.session_state.messages.append(
                    {
                        "role": "function",
                        "content": "[Image]" if res_type == "image" else res,  # 调用函数返回结果
                    }
                )

                break

        params["messages"] = st.session_state.messages
        response = client.chat.completions.create(**params)


def main():
    st.title("💬 Code Interpreter")

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("INTERPRETER_CHAT_API_BASE"),
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role = message["role"]
        if role in ["user", "function"]:
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(postprocess_text(message["content"]))

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            chat_once(message_placeholder, client)


if __name__ == "__main__":
    main()
