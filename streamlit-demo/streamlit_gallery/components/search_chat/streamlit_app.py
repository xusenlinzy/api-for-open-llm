import os

import streamlit as st
from langchain_community.utilities import SerpAPIWrapper
from openai import OpenAI

PROMPT_TEMPLATE = """<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>

<已知信息>问题的搜索结果为：{context}</已知信息>

<问题>{query}</问题>"""


def main():
    st.title("💬 Search Chatbot")

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("CHAT_API_BASE"),
    )

    search = SerpAPIWrapper()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message["reference"] is not None:
                st.markdown("### Reference Search Results")
                st.json(message["reference"], expanded=False)

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = search.run(prompt)
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model="baichuan",
                messages=[
                     {
                         "role": m["role"],
                         "content": m["content"]
                     }
                     for m in st.session_state.messages[:-1]
                 ] + [
                     {
                         "role": "user",
                         "content": PROMPT_TEMPLATE.format(query=prompt, context=result)
                     }
                 ],
                max_tokens=st.session_state.get("max_tokens", 512),
                temperature=st.session_state.get("temperature", 0.9),
                stream=True,
            ):
                full_response += response.choices[0].delta.content or ""

                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.markdown("### Reference Search Results")
            st.json({"search_result": result}, expanded=False)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "reference": {"search_result": result},
            }
        )


if __name__ == "__main__":
    main()
