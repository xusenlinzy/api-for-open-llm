from typing import List, Optional

from langchain.chains.sql_database.prompt import SQL_PROMPTS
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

answer_prompt = PromptTemplate.from_template(
    """给出以下用户问题、相应的 SQL 查询和 SQL 结果，请回答用户问题。

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)


def create_sql_query(
    query: str,
    base_url: str,
    database_uri: str,
    include_tables: Optional[List[str]] = None,
    sample_rows_in_table_info: Optional[int] = 1,
):
    question = {"question": query}

    db = SQLDatabase.from_uri(
        database_uri,
        include_tables=include_tables,
        sample_rows_in_table_info=sample_rows_in_table_info,
    )

    llm = ChatOpenAI(
        model="codeqwen",
        temperature=0,
        openai_api_base=base_url,
        openai_api_key="xxx"
    )

    prompt_to_use = SQL_PROMPTS[db.dialect]
    if {"input", "top_k", "table_info"}.difference(prompt_to_use.input_variables):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    }
    write = (
        RunnablePassthrough.assign(**inputs)  # type: ignore
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(5))
        | llm.bind(stop=["\nSQLResult:", "SQLResult"])
        | StrOutputParser()
    )

    sql_query = write.invoke(question).strip()
    sql_result = db.run(sql_query, fetch="cursor")

    return sql_query, sql_result


def create_llm_chain(base_url: str):
    llm = ChatOpenAI(
        model="codeqwen",
        temperature=0,
        openai_api_base=base_url,
        openai_api_key="xxx"
    )
    return answer_prompt | llm | StrOutputParser()


if __name__ == "__main__":
    import pandas as pd

    sql_query, sql_result = create_sql_query(
        "2024年各个信息来源分别发布了多少资讯,按照数量排序",
        base_url="http://192.168.20.44:7861/v1",
        include_tables=["document", "source"],
        database_uri="mysql+pymysql://root:Dnect_123@192.168.0.52:3306/information_services",
    )
    print(pd.DataFrame(sql_result))
