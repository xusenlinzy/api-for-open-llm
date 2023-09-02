import pandas as pd
from func_timeout import func_timeout
from sqlalchemy import create_engine


def query_table_names(db_name: str, db_creds: dict, timeout: float = 10.0) -> list:
    """ 获取数据库表名 """
    engine = None
    try:
        db_url = f"mysql+pymysql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}"
        engine = create_engine(db_url)
        tables = func_timeout(timeout, pd.read_sql_query, args=("show tables;", engine))
        engine.dispose()  # close connection
        return [t[0] for t in tables.values]
    except Exception as e:
        if engine:
            engine.dispose()  # close connection if query fails/timeouts
        raise e


def query_table_schema(
    table_name: str, db_name: str, db_creds: dict, timeout: float = 10.0
):
    """ 获取数据库表结构 """
    engine = None
    try:
        db_url = f"mysql+pymysql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}"
        engine = create_engine(db_url)
        schema = func_timeout(
            timeout, pd.read_sql_query, args=(f"show create table {table_name};", engine)
        )["Create Table"][0]
        engine.dispose()  # close connection
        return schema
    except Exception as e:
        if engine:
            engine.dispose()  # close connection if query fails/timeouts
        raise e


def query_mysql_db(
    query: str, db_name: str, db_creds: dict, timeout: float = 10.0
):
    """ 获取数据库查询结果 """
    engine = None
    try:
        db_url = f"mysql+pymysql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}"
        engine = create_engine(db_url)
        results_df = func_timeout(timeout, pd.read_sql_query, args=(query, engine))
        engine.dispose()  # close connection
        return results_df
    except Exception as e:
        if engine:
            engine.dispose()  # close connection if query fails/timeouts
        raise e


SQL_PROMPT = """### Instructions:
Your task is to convert a question into a SQL query, given a database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{query}`.
This query will run on a database whose schema is represented in this string:
{table_info}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{query}`:
```sql
"""
