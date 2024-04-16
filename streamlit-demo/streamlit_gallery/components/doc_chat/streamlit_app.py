import os
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from .utils import DocServer, DOCQA_PROMPT


def main():
    UPLOAD_FOLDER = os.path.join(Path(__file__).parents[3], "upload_files")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("CHAT_API_BASE"),
    )

    @st.cache_resource
    def load_doc_server():
        embeddings = OpenAIEmbeddings(
            openai_api_base=os.getenv("EMBEDDING_API_BASE"),
            openai_api_key=os.getenv("API_KEY", ""),
        )
        server = DocServer(embeddings)
        return server

    server = load_doc_server()

    @st.cache_resource
    def create_index(file, chunk_size, chunk_overlap):
        filename = file.name
        filepath = f"{UPLOAD_FOLDER}/{filename}"
        with open(filepath, "wb") as f:
            f.write(file.read())

        file_id = server.upload(filepath, chunk_size, chunk_overlap)
        st.session_state.update(dict(file_id=file_id))

        os.remove(filepath)
        return file.name

    def delete_index(file_id):
        server.delete(file_id)
        return file_id

    st.title("💬 Document Chatbot")

    with st.expander("📚‍ FILES", False):
        file = st.file_uploader("Upload file", accept_multiple_files=False)
        if file:
            file_name = create_index(
                file,
                chunk_size=st.session_state.get("chunk_size", 250),
                chunk_overlap=st.session_state.get("chunk_overlap", 50),

            )
            st.session_state.update(dict(file_name=file_name))

        vector_store_names = server.db.table_names()
        vector_store_name = st.selectbox("Select a vector store", vector_store_names)
        st.session_state.update(dict(vector_store_name=vector_store_name))

        if st.button("❌️ DELETE FILE"):
            _ = delete_index(vector_store_name)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and isinstance(message["reference"], pd.DataFrame):
                with st.expander(label="### Reference Documents"):
                    st.dataframe(message["reference"])

    if prompt := st.chat_input("What is up?"):
        vector_store_name = st.session_state.get("vector_store_name", None)
        doc_prompt, reference = None, None
        if vector_store_name is not None:
            result = server.search(
                query=prompt,
                top_k=st.session_state.get("top_k", 3),
                table_name=vector_store_name,
                rerank=st.session_state.get("rerank", False),
            )

            context = "\n\n".join(doc for doc in result["text"].tolist())
            doc_prompt = DOCQA_PROMPT.format(query=prompt, context=context)
            reference = result

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            pyload = dict(
                model="qwen2",
                messages=[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages[:-1]
                ] + [
                        {
                            "role": "user",
                            "content": doc_prompt or prompt
                        }
                ],
                stream=True,
                max_tokens=st.session_state.get("max_tokens", 512),
                temperature=st.session_state.get("temperature", 0.9),
            )

            for response in client.chat.completions.create(**pyload):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            if isinstance(reference, pd.DataFrame):
                with st.expander(label="### Reference Documents"):
                    st.dataframe(reference)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "reference": reference,
            }
        )


if __name__ == "__main__":
    main()
