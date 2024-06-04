import streamlit as st
from openai import OpenAI


def main():
    st.title("💬 Chatbot")

    client = OpenAI(
        api_key=st.session_state.get("api_key", "xxx"),
        base_url=st.session_state.get("base_url", "xxx"),
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
                model=st.session_state.get("model_name", "xxx"),
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
