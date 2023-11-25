import json
import os

import streamlit as st
from openai import OpenAI

from .utils import functions, available_functions, postprocess_text


def chat_once(functions, message_placeholder, client: OpenAI):
    params = dict(
        model="chatglm3",
        messages=st.session_state.messages,
        stream=True,
        functions=functions,
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
                    function_call = chunk.choices[0].delta.function_call
                    st.info(f"**Function Call Response ==>** {function_call.model_dump()}")

                    function_to_call = available_functions[function_call.name]
                    function_args = json.loads(function_call.arguments)
                    tool_response = function_to_call(**function_args)
                    st.info(f"**Tool Call Response ==>** {tool_response}")
                except:
                    continue

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "function_call": function_call,
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
        response = client.chat.completions.create(**params)


def main():
    st.title("💬 Tool Chatbot")

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("TOOL_CHAT_API_BASE"),
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
            chat_once(functions, message_placeholder, client)


if __name__ == "__main__":
    main()
