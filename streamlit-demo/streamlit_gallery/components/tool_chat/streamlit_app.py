import json
import os

import openai
import streamlit as st

from .utils import functions, available_functions


def chat_once(functions, message_placeholder):
    params = dict(
        model="chatglm3",
        messages=st.session_state.messages,
        stream=True,
        functions=functions,
        max_tokens=st.session_state.get("max_tokens", 512),
        temperature=st.session_state.get("temperature", 0.9),
    )
    response = openai.ChatCompletion.create(**params)

    display = ""
    for _ in range(5):
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.get("content", "")
            full_response += content
            display += content
            message_placeholder.markdown(display + "▌")

            if chunk.choices[0].finish_reason == "stop":
                message_placeholder.markdown(display + "▌")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response
                    }
                )
                return

            elif chunk.choices[0].finish_reason == "function_call":
                try:
                    function_call = chunk.choices[0].delta.function_call
                    st.info(f"**Function Call Response ==>** {function_call.to_dict_recursive()}")

                    function_to_call = available_functions[function_call.name]
                    function_args = json.loads(function_call.arguments)
                    tool_response = function_to_call(**function_args)
                    st.info(f"**Tool Call Response ==>** {tool_response}")
                except:
                    continue

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response
                    }
                )
                st.session_state.messages.append(
                    {
                        "role": "function",
                        "name": function_call.name,
                        "content": tool_response,  # 调用函数返回结果
                    }
                )

                break

        params["messages"] = st.session_state.messages
        response = openai.ChatCompletion.create(**params)


def main():
    st.title("💬 Chatbot")

    openai.api_base = os.getenv("TOOL_CHAT_API_BASE", "http://192.168.20.59:7891/v1")
    openai.api_key = os.getenv("API_KEY", "xxx")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role = message["role"]
        if role in ["user", "function"]:
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            chat_once(functions, message_placeholder)


if __name__ == "__main__":
    main()
