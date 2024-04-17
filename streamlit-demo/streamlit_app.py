import os

import streamlit as st

from streamlit_gallery.utils.page import page_group


def main():
    from streamlit_gallery.apps import gallery
    from streamlit_gallery.components import chat, doc_chat

    page = page_group("p")

    with st.sidebar:
        st.title("🎉 LLM Gallery")

        with st.expander("✨ APPS", True):
            page.item("LLM Chat Gallery", gallery, default=True)

        with st.expander("🧩 COMPONENTS", True):

            if os.getenv("CHAT_API_BASE", ""):
                page.item("Chat", chat)
                page.item("Doc Chat", doc_chat)

            if os.getenv("SQL_CHAT_API_BASE", ""):
                from streamlit_gallery.components import sql_chat
                page.item("SQL Chat", sql_chat)

            if os.getenv("SERPAPI_API_KEY", ""):
                from streamlit_gallery.components import search_chat
                page.item("Search Chat", search_chat)

            if os.getenv("TOOL_CHAT_API_BASE", ""):
                from streamlit_gallery.components import tool_chat
                page.item("Tool Chat", tool_chat)

            if os.getenv("INTERPRETER_CHAT_API_BASE", ""):
                from streamlit_gallery.components import code_interpreter
                page.item("Code Interpreter", code_interpreter)

        with st.expander("🐧 PARAMTERS", False):
            max_tokens = st.slider("MaxTokens", 20, 4096, 1024)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.9)
            chunk_size = st.slider("ChunkSize", 100, 512, 250)
            chunk_overlap = st.slider("CHUNK_OVERLAP", 0, 100, 50)
            top_k = st.slider("Top_K", 0, 10, 4)
            rerank = st.checkbox("Apply Rerank")

            st.session_state.update(
                dict(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    rerank=rerank,
                )
            )

        if st.button("🗑️ CLEAR MESSAGES"):
            st.session_state.messages = []

    page.show()


if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit LLM Gallery", page_icon="🎈", layout="wide")
    main()
