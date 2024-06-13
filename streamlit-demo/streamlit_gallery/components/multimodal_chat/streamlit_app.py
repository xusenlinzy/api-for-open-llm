import base64

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
        if message["role"] == "user":
            with st.chat_message(name="user", avatar="user"):
                st.markdown(message["content"])
        else:
            with st.chat_message(name="model", avatar="assistant"):
                st.markdown(message["content"])

    uploaded_image = st.sidebar.file_uploader(
        "Upload image",
        key=1,
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    with st.sidebar:
        if uploaded_image is not None:
            st.image(uploaded_image)

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }

            if uploaded_image:
                base64_image = base64.b64encode(uploaded_image.read()).decode('utf-8')
                msg["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{uploaded_image.type};base64,{base64_image}"
                        }
                    }
                )

            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state.get("model_name", "xxx"),
                messages=[msg],
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
