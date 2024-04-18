import os

import pandas as pd
import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase

from .utils import create_sql_query, create_llm_chain


def main():
    st.title("💬 SQL Chatbot")

    base_url = os.getenv("SQL_CHAT_API_BASE")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander(label="✨ 简介"):
            st.markdown("""+ SQL问答流程：

    + 基于用户问题和选定表结构生成可执行的 sql 语句

    + 执行 sql 语句，返回数据库查询结果
    
    + [TODO] 通过 schema link 自动寻找相关的表

    + [TODO] 根据查询结果对用户问题进行回复""")

    with col2:
        with st.expander("🐬 数据库配置", False):
            db_url = st.text_input("URL", placeholder="mysql+pymysql://")
            if db_url:
                try:
                    db = SQLDatabase.from_uri(database_uri=db_url)
                    table_names = db.get_usable_table_names()
                except:
                    table_names = []
                    st.error("Wrong configuration for database connection!")

                include_tables = st.multiselect("选择查询表", table_names)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])
                st.markdown("### SQL Query")
                if message["sql"] is not None:
                    st.code(message["sql"], language="sql")
                if message["data"] is not None:
                    with st.expander("展示查询结果"):
                        st.dataframe(message["data"], use_container_width=True)

    if query := st.chat_input("2022年xx大学参与了哪些项目？"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            sql_query, sql_result = create_sql_query(query, base_url, db_url, include_tables)
            data = pd.DataFrame(sql_result) if sql_result else None
            str_data = data.to_markdown() if data is not None else ""

            llm_chain = create_llm_chain(base_url)
            for chunk in llm_chain.stream(
                {"question": query, "query": sql_query, "result": str_data}
            ):
                full_response += chunk or ""
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            if sql_query:
                st.markdown("### SQL Query")
                st.code(sql_query, language="sql")

            if data is not None:
                with st.expander("展示查询结果"):
                    st.dataframe(data, use_container_width=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sql": sql_query,
                "data": data,
            }
        )


if __name__ == "__main__":
    main()
