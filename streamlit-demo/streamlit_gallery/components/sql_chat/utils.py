from typing import List, Optional

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

PROMPT_SUFFIX = """只能使用以下表格：
{table_info}

问题：
{input}

包含SQL查询的JSON返回结果为：
"""

query_prompt = """您是一个SQL专家。给定一个输入问题，请按照以下要求生成一个语法正确的SQL查询来运行。
1. 除非用户在问题中指定了要获取的示例的具体数量，否则最多只能查询{top_k}个结果，并按照SQL的规定使用LIMIT子句。您可以对结果进行排序，以返回数据库中信息量最大的数据。
2. 切勿查询表中的所有列。必须只查询回答问题所需的列。用反斜线 (`) 包住每一列的名称，将其表示为分隔标识符。
3. 注意只使用在下表中可以看到的列名。注意不要查询不存在的列。此外，还要注意哪一列在哪个表中。
4. 如果问题涉及 “今天”，请注意使用 CURDATE() 函数获取当前日期。
5. 返回结果必须为JSON格式，其仅有一个属性`sql`，值为SQL查询语句，请不要返回多余的任何信息。

"""


def extract_json(json_: str):
    from langchain.output_parsers.json import SimpleJsonOutputParser

    parser = SimpleJsonOutputParser()
    try:
        return parser.parse(json_)
    except:
        return None


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

    prompt_to_use = PromptTemplate(
        input_variables=["input", "table_info", "dialect", "top_k"],
        template=query_prompt + PROMPT_SUFFIX,
    )
    if {"input", "top_k", "table_info"}.difference(prompt_to_use.input_variables):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    inputs = {
        "input": lambda x: x["question"],
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
        | llm
        | StrOutputParser()
    )

    sql_query = extract_json(write.invoke(question).strip())
    if sql_query is not None:
        sql_query = sql_query["sql"]
        sql_result = db.run(sql_query, fetch="cursor")
    else:
        sql_result = "SQLResult is not correctly queried!"

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
        base_url="http://192.168.0.59:7891/v1",
        include_tables=["document", "source"],
        database_uri="mysql+pymysql://root:Dnect_123@192.168.0.52:3306/information_services",
    )
    print(pd.DataFrame(sql_result))
