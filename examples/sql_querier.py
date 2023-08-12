import json
import sqlite3

import openai
from loguru import logger


def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results


class SqlQuerier:
    def __init__(self, openai_api_base, openai_api_key="xxx", db_path="Chinook.db"):
        openai.api_base = openai_api_base
        openai.api_key = openai_api_key
        self.conn = sqlite3.connect(db_path)
        logger.info("Opened database successfully")

    def run(self, query, database_schema):
        # Step 1: send the conversation and available functions to model
        messages = [{"role": "user", "content": query}]
        functions = [
            {
                "name": "ask_database",
                "description": "该工具用来回答音乐相关的问题，输出应该是一个标准化的SQL查询语句。",
                "parameters": [
                    {
                        'name': 'query',
                        'description': f"基于下面数据库表结构的SQL查询语句，用来回答用户问题。\n\n{database_schema}",
                        'required': True,
                        'schema': {
                            'type': 'string'
                        },
                    },
                ],
            }
        ]
        response = openai.ChatCompletion.create(
            model="qwen",
            messages=messages,
            temperature=0,
            functions=functions,
            stop=["Observation:"]
        )

        logger.info(response["choices"][0]["message"]["function_call"])

        answer = ""
        response_message = response["choices"][0]["message"]
        # Step 2: check if model wanted to call a function
        if response_message.get("function_call"):
            logger.info(f"Function call: {response_message['function_call']}")
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "ask_database": ask_database,
            }  # only one function in this example
            function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            logger.info(f"Function args: {function_args}")

            function_response = fuction_to_call(self.conn, function_args["query"])
            logger.info(f"Function response: {function_response}")

            # Step 4: send the info on the function call and function response to model
            messages.append(response_message)  # extend conversation with assistant's reply
            messages.append(
                {
                    "role": "function",
                    "content": function_response,
                }
            )  # extend conversation with function response

            second_response = openai.ChatCompletion.create(
                model="qwen",
                messages=messages,
                temperature=0,
                functions=functions,
            )  # get a new response from model where it can see the function response
            answer = second_response["choices"][0]["message"]["content"]
            logger.info(f"Model output: {answer}")

            j = answer.rfind("Final Answer:")
            answer = answer[j + 14:] if answer else answer

        return answer


if __name__ == '__main__':
    database_schema = """create table albums
AlbumId INTEGER not null primary key autoincrement, --专辑ID
Title NVARCHAR(160) not null, --专辑名称
ArtistId INTEGER not null references artists --艺术家ID
);
"""

    openai_api_base = "http://192.168.0.53:7891/v1"
    query = "发行专辑最多的艺术家是谁？"
    sql_querier = SqlQuerier(openai_api_base)
    answer = sql_querier.run(query, database_schema)
    print(answer)
