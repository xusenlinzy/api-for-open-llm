import os

import streamlit as st
from openai import OpenAI


def main():
    st.title("💬 Chatbot")

    with st.expander(label="Note"):
        st.code("""普通问答是指使用大型语言模型进行的常见问题解答过程。
这种类型的问答通常涉及用户提出一个或多个问题，而模型基于其训练数据来提供最可能的答案或解释。""")

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("CHAT_API_BASE"),
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model="baichuan",
                messages=[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages
                ],
                max_tokens=st.session_state.get("max_tokens", 512),
                temperature=st.session_state.get("temperature", 0.9),
                stream=True,
            ):
                full_response += response.choices[0].delta.content or ""

                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response
            }
        )


if __name__ == "__main__":
    main()
