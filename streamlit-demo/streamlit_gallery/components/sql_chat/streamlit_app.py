import os

import openai
import streamlit as st

from .utils import query_table_names, query_table_schema, query_mysql_db


@st.cache_resource
def get_table_names(db_name, db_creds):
    return query_table_names(db_name, db_creds)


@st.cache_resource
def get_table_info(table_name, db_name, db_creds):
    return query_table_schema(table_name, db_name, db_creds)


def main():
    st.title("💬 SQL Chatbot")

    openai.api_base = os.getenv("SQL_CHAT_API_BASE")
    openai.api_key = os.getenv("API_KEY")

    with st.expander("🐬 DATABASE SETTINGS", False):
        db_host = st.text_input("Host", value="192.168.0.121")
        db_port = st.number_input("Port", value=3306)
        db_user = st.text_input("User", value="root")
        db_password = st.text_input("Password", type="password", value="xxxx")
        db_name = st.text_input("Database Name", value="test2")

        db_creds = dict(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
        )

        if db_name and db_creds:
            table_names = get_table_names(db_name, db_creds)
            table_name = st.selectbox("Select a table", table_names)
            st.session_state.update(dict(table_name=table_name))

            table_info = get_table_info(table_name, db_name, db_creds)
            st.session_state.update(dict(table_info=table_info))

        st.session_state.update(dict(db_creds=db_creds, db_name=db_name))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("2022年xx大学参与了哪些项目？"):
        table_name = st.session_state.get("table_name", None)
        if table_name:
            prompt = prompt.format(query=prompt, table_info=st.session_state.get("table_info", ""))

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for response in openai.Completion.create(
                model="sqlcoder",
                prompt=prompt,
                stream=True,
                temperature=0.0,
            ):
                full_response += response.choices[0].text
                message_placeholder.code(full_response + "▌", language="sql")

            message_placeholder.code(full_response, language="sql")
            result = query_mysql_db(
                full_response,
                db_name=st.session_state.get("db_name"),
                db_creds=st.session_state.get("db_creds"),
            )

            if result is not None:
                st.dataframe(result.head(5))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response
            }
        )


if __name__ == "__main__":
    main()
