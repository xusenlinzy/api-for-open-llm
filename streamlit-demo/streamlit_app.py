import streamlit as st

from streamlit_gallery.utils.page import page_group


def main():
    from streamlit_gallery.apps import gallery
    from streamlit_gallery.components import chat, doc_chat, sql_chat, search_chat

    page = page_group("p")

    with st.sidebar:
        st.title("🎉 Morris's Gallery")

        with st.expander("✨ APPS", True):
            page.item("Morris Chat Gallery", gallery, default=True)

        with st.expander("🧩 COMPONENTS", True):
            page.item("Chat", chat)
            page.item("Doc Chat", doc_chat)
            page.item("SQL Chat", sql_chat)
            page.item("Search Chat", search_chat)

        with st.expander("🐧 PARAMTERS", False):
            max_tokens = st.slider("MaxTokens", 20, 4096, 1024)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.9)
            chunk_size = st.slider("ChunkSize", 100, 512, 300)
            chunk_overlap = st.slider("CHUNK_OVERLAP", 0, 100, 10)
            top_k = st.slider("Top_K", 0, 10, 3)

            st.session_state.update(
                dict(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )
            )

        if st.button("🗑️ CLEAR MESSAGES"):
            st.session_state.messages = []

    page.show()


if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Morris", page_icon="🎈", layout="wide")
    main()
