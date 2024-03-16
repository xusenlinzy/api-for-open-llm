import os
import shutil
from pathlib import Path

from openai import OpenAI

import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings

from .utils import FaissDocServer, Embeddings, DOCQA_PROMPT


def main():
    UPLOAD_FOLDER = os.path.join(Path(__file__).parents[3], "upload_files")
    VECTOR_STORE_PATH = os.path.join(Path(__file__).parents[3], "vector_stores")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("CHAT_API_BASE"),
    )

    @st.cache_resource
    def load_doc_server():
        embedding_name = os.getenv("EMBEDDING_NAME", "")
        if embedding_name:
            embedding = Embeddings(embedding_name)
        else:
            embedding = OpenAIEmbeddings(
                openai_api_base=os.getenv("EMBEDDING_API_BASE"),
                openai_api_key=os.getenv("API_KEY", ""),
            )
        server = FaissDocServer(embedding)
        return server

    FAISS = load_doc_server()

    @st.cache_resource
    def create_index(file, chunk_size, chunk_overlap):
        filename = file.name
        file_path = f"{UPLOAD_FOLDER}/{filename}"
        with open(file_path, "wb") as f:
            f.write(file.read())

        vs_path = f"{VECTOR_STORE_PATH}/{filename}"
        FAISS.doc_upload(file_path, chunk_size, chunk_overlap, vs_path)
        return file.name

    def delete_index(filename):
        file_path = f"{UPLOAD_FOLDER}/{filename}"
        os.remove(file_path)
        vs_path = f"{VECTOR_STORE_PATH}/{filename}"
        shutil.rmtree(vs_path)
        return filename

    st.title("💬 Document Chatbot")


    with st.expander("📚‍ FILES", False):
        file = st.file_uploader("Upload file", accept_multiple_files=False)
        if file:
            file_name = create_index(
                file,
                chunk_size=st.session_state.get("chunk_size", 300),
                chunk_overlap=st.session_state.get("chunk_overlap", 10),
            )
            st.session_state.update(dict(file_name=file_name))

        vector_store_names = os.listdir(VECTOR_STORE_PATH)
        vector_store_name = st.selectbox("Select a vector store", vector_store_names)
        st.session_state.update(
            dict(vector_store_name=vector_store_name)
        )

        if st.button("❌️ DELETE FILE"):
            _ = delete_index(vector_store_name)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message["reference"] is not None:
                st.markdown("### Reference Documents")
                st.json(message["reference"], expanded=False)

    if prompt := st.chat_input("What is up?"):
        vector_store_name = st.session_state.get("vector_store_name", None)
        doc_prompt, reference = None, None
        if vector_store_name is not None:
            result = FAISS.doc_search(
                query=prompt,
                top_k=st.session_state.get("top_k", 3),
                vs_path=f"{VECTOR_STORE_PATH}/{vector_store_name}"
            )
            context = "\n".join([doc[0].page_content for doc in result])
            doc_prompt = DOCQA_PROMPT.format(query=prompt, context=context)
            reference = [
                {
                    "content": doc[0].page_content,
                    "score": float(doc[1])
                }
                for doc in result
            ]

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            pyload = dict(
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
            if reference is not None:
                st.markdown("### Reference Documents")
                st.json(reference, expanded=False)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "reference": reference,
            }
        )


if __name__ == "__main__":
    main()
