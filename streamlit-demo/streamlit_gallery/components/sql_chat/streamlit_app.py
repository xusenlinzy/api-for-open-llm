import os

import openai
import streamlit as st

from .utils import query_table_names, query_table_schema, query_mysql_db, SQL_PROMPT


def main():
    st.title("💬 SQL Chatbot")

    openai.api_base = os.getenv("SQL_CHAT_API_BASE")
    openai.api_key = os.getenv("API_KEY")

    with st.expander("🐬 DATABASE SETTINGS", False):
        col1, col2 = st.columns(2)
        with col1:
            db_host = st.text_input("Host", placeholder="192.168.0.121")
            db_user = st.text_input("User", value="root")
            db_name = st.text_input("Database Name", placeholder="test2")
        with col2:
            db_port = st.number_input("Port", value=3306)
            db_password = st.text_input("Password", type="password")

        db_creds = dict(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
        )

        if db_name and db_creds:
            with col2:
                table_names = query_table_names(db_name, db_creds)
                table_name = st.selectbox("Select a table", table_names)
            st.session_state.update(dict(table_name=table_name))

            table_info = query_table_schema(table_name, db_name, db_creds)
            st.session_state.update(dict(table_info=table_info))

        st.session_state.update(dict(db_creds=db_creds, db_name=db_name))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.code(message["content"])
                if message["result"] is not None:
                    st.dataframe(message["result"])

    if prompt := st.chat_input("2022年xx大学参与了哪些项目？"):
        table_name = st.session_state.get("table_name", None)
        sql_prompt = None
        if table_name:
            sql_prompt = SQL_PROMPT.format(query=prompt, table_info=st.session_state.get("table_info", ""))

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for response in openai.Completion.create(
                model="sqlcoder",
                prompt=sql_prompt or prompt,
                stream=True,
                temperature=0.0,
                stop=["```"],
            ):
                full_response += response.choices[0].text
                message_placeholder.code(full_response + "▌", language="sql")

            message_placeholder.code(full_response, language="sql")
            try:
                result = query_mysql_db(
                    full_response,
                    db_name=st.session_state.get("db_name"),
                    db_creds=st.session_state.get("db_creds"),
                )
            except:
                result = None

            if result is not None:
                result = result.head(5)
                st.dataframe(result)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "result": result,
            }
        )


if __name__ == "__main__":
    main()
