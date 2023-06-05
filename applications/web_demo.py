import logging
import os
import re
import shutil

import gradio as gr
import openai
import pandas as pd
from backoff import on_exception, expo
from sqlalchemy import create_engine

from tools.doc_qa import DocQAPromptAdapter
from tools.web.overwrites import postprocess, reload_javascript
from tools.web.presets import (
    small_and_beautiful_theme,
    title,
    description,
    description_top,
    CONCURRENT_COUNT
)
from tools.web.utils import (
    convert_to_markdown,
    shared_state,
    reset_textbox,
    cancel_outputing,
    transfer_input,
    reset_state,
    delete_last_conversation
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


openai.api_key = "xxx"
doc_adapter = DocQAPromptAdapter()


def add_llm(model_name, api_base, models):
    """ 添加模型 """
    models = models or {}
    if model_name and api_base:
        models.update(
            {
                model_name: api_base
            }
        )
    choices = [m[0] for m in models.items()]
    return "",  "", models, gr.Dropdown.update(choices=choices, value=choices[0] if choices else None)


def set_openai_env(api_base):
    """ 配置接口地址 """
    openai.api_base = api_base
    doc_adapter.embeddings.openai_api_base = api_base


def get_file_list():
    """ 获取文件列表 """
    if not os.path.exists("doc_store"):
        return []
    return os.listdir("doc_store")


file_list = get_file_list()


def upload_file(file):
    """ 上传文件 """
    if not os.path.exists("doc_store"):
        os.mkdir("doc_store")

    if file is not None:
        filename = os.path.basename(file.name)
        shutil.move(file.name, f"doc_store/{filename}")
        file_list = get_file_list()
        file_list.remove(filename)
        file_list.insert(0, filename)
        return gr.Dropdown.update(choices=file_list, value=filename)


def add_vector_store(filename, model_name, models, chunk_size, chunk_overlap):
    """ 将文件转为向量数据存储 """
    api_base = models[model_name]
    set_openai_env(api_base)
    doc_adapter.chunk_size = chunk_size
    doc_adapter.chunk_overlap = chunk_overlap

    if filename is not None:
        vs_path = f"vector_store/{filename.split('.')[0]}-{filename.split('.')[-1]}"
        if not os.path.exists(vs_path):
            doc_adapter.create_vector_store(f"doc_store/{filename}", vs_path=vs_path)
            msg = f"Successfully added vector store for {filename}!"
        else:
            doc_adapter.reset_vector_store(vs_path=vs_path)
            msg = f"Successfully loaded vector store for {filename}!"
    else:
        msg = "Please select a file!"
    return msg


def add_db(db_user, db_password, db_host, db_port, db_name, databases):
    """ 添加数据库 """
    databases = databases or {}
    if db_user and db_password and db_host and db_port and db_name:
        databases.update(
            {
                db_name: {
                    "user": db_user,
                    "password": db_password,
                    "host": db_host,
                    "port": int(db_port)
                }
            }
        )
    choices = [m[0] for m in databases.items()]
    return "", "", "", "", "", databases, gr.Dropdown.update(choices=choices, value=choices[0] if choices else None)


def get_table_names(select_database, databases):
    """ 获取数据库表名 """
    if select_database:
        db_config = databases[select_database]
        con = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{select_database}")
        tables = pd.read_sql("show tables;", con=con).values
        tables = [t[0] for t in tables]
        return gr.Dropdown.update(choices=tables, value=[tables[0]])


def get_sql_result(x, con):
    q = r"sql\n(.+?);\n"
    sql = re.findall(q, x, re.DOTALL)[0] + ";"
    df = pd.read_sql(sql, con=con).iloc[:50, :]
    return df


@on_exception(expo, openai.error.RateLimitError, max_tries=5)
def chat_completions_create(params):
    """ chat接口 """
    return openai.ChatCompletion.create(**params)


def predict(
    model_name,
    models,
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_tokens,
    memory_k,
    is_kgqa,
    single_turn,
    is_dbqa,
    select_database,
    select_table,
    databases,
):
    api_base = models[model_name]
    set_openai_env(api_base)

    if text == "":
        yield chatbot, history, "Empty context.", None
        return

    if history is None:
        history = []

    messages = []
    if is_dbqa:
        temperature = 0.0
        db_config = databases[select_database]
        con = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{select_database}")
        table_schema = ""
        for t in select_table:
            table_schema += pd.read_sql(f"show create table {t};", con=con)["Create Table"][0] + "\n\n"
        table_schema = table_schema.replace("DEFAULT NULL", "")
        messages.append(
            {
                "role": "system",
                "content": f"你现在是一名SQL助手，能够根据用户的问题生成准确的SQL查询。已知SQL的建表语句为：{table_schema}根据上述数据库信息，回答相关问题。"
            },
        )
    else:
        if not single_turn:
            for h in history[-memory_k:]:
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": h[0]
                        },
                        {
                            "role": "assistant",
                            "content": h[1]
                        }
                    ]
                )

    messages.append(
        {
            "role": "user",
            "content": doc_adapter(text) if is_kgqa else text
        }
    )

    params = dict(
        stream=True,
        messages=messages,
        model=model_name,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens
    )

    res = chat_completions_create(params)
    x = ""
    for openai_object in res:
        delta = openai_object.choices[0]["delta"]
        if "content" in delta:
            x += delta["content"]

        a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
            [text, convert_to_markdown(x)]
        ], history + [[text, x]]

        yield a, b, "Generating...", None

    if shared_state.interrupted:
        shared_state.recover()
        try:
            yield a, b, "Stop: Success", None
            return
        except:
            pass

    try:
        if is_dbqa:
            df = get_sql_result(x, con)
            yield a, b, "Generate: Success", df
        else:
            yield a, b, "Generate: Success", None
    except:
        pass


def retry(
    model_name,
    models,
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_tokens,
    memory_k,
    is_kgqa,
    single_turn,
    is_dbqa,
    select_database,
    select_table,
    databases,
):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, "Empty context."
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict(
        model_name,
        models,
        inputs,
        chatbot,
        history,
        top_p,
        temperature,
        max_tokens,
        memory_k,
        is_kgqa,
        single_turn,
        is_dbqa,
        select_database,
        select_table,
        databases,
    ):
        yield x


gr.Chatbot.postprocess = postprocess

with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    history = gr.State([])
    user_question = gr.State("")

    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")

    gr.Markdown(description_top)

    with gr.Tab("Generate"):
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=5):
                with gr.Row():
                    chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height="100%")
                with gr.Row():
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(
                            show_label=False, placeholder="Enter text"
                        ).style(container=False)
                    with gr.Column(min_width=70, scale=1):
                        submitBtn = gr.Button("发送")
                    with gr.Column(min_width=70, scale=1):
                        cancelBtn = gr.Button("停止")
                with gr.Row():
                    emptyBtn = gr.Button(
                        "🧹 新的对话",
                    )
                    retryBtn = gr.Button("🔄 重新生成")
                    delLastBtn = gr.Button("🗑️ 删除最旧对话")

            with gr.Column():
                with gr.Column(min_width=50, scale=1):
                    with gr.Tab(label="模型"):
                        model_name = gr.Textbox(
                            placeholder="chatglm",
                            label="模型名称",
                        )
                        api_base = gr.Textbox(
                            placeholder="https://0.0.0.0:80/v1",
                            label="模型接口地址",
                        )
                        add_model = gr.Button(
                            value="\U0001F31F 添加模型",
                        )
                        with gr.Accordion(open=False, label="所有模型配置"):
                            # models = gr.Json()
                            models = gr.Json(
                                value={
                                    "chatglm": "http://192.168.0.59:80/v1",
                                    "moss": "http://192.168.0.58:80/v1"
                                }
                            )
                        single_turn = gr.Checkbox(label="使用单轮对话", value=False)
                        select_model = gr.Dropdown(
                            choices=[m[0] for m in models.value.items()] if models.value else [],
                            value=[m[0] for m in models.value.items()][0] if models.value else None,
                            label="选择模型",
                            interactive=True,
                            elem_classes="llm-selector",
                        )

                    with gr.Tab(label="知识库"):
                        is_kgqa = gr.Checkbox(
                            label="使用知识库问答",
                            value=False,
                            interactive=True,
                        )
                        gr.Markdown("""**基于本地知识库生成更加准确的回答！**""")
                        select_file = gr.Dropdown(
                            choices=file_list,
                            label="选择文件",
                            interactive=True,
                            value=file_list[0] if len(file_list) > 0 else None,
                            elem_classes="llm-selector",
                        )
                        file = gr.File(
                            label="上传文件",
                            visible=True,
                            file_types=['.txt', '.md', '.docx', '.pdf']
                        )
                        add_vs = gr.Button(value="📖 添加到知识库")

                    with gr.Tab(label="数据库"):
                        with gr.Accordion(open=False, label="数据库配置"):
                            with gr.Row():
                                db_name = gr.Textbox(
                                    placeholder="test",
                                    label="数据库名称",
                                )
                            with gr.Row():
                                db_user = gr.Textbox(
                                    placeholder="root",
                                    label="用户名",
                                )
                                db_password = gr.Textbox(
                                    placeholder="password",
                                    label="密码",
                                    type="password"
                                )
                            with gr.Row():
                                db_host = gr.Textbox(
                                    placeholder="0.0.0.0",
                                    label="主机",
                                )
                                db_port = gr.Textbox(
                                    placeholder="3306",
                                    label="端口",
                                )
                        add_database = gr.Button("🐬 添加数据库")

                        with gr.Accordion(open=False, label="所有数据库配置"):
                            # databases = gr.Json()
                            databases = gr.Json(
                                value={
                                    "complaint_database": {
                                        "user": "live_monitor",
                                        "password": "live_monitor",
                                        "host": "192.168.0.13",
                                        "port": 3306
                                    }
                                }
                            )
                        select_database = gr.Dropdown(
                            choices=[d[0] for d in databases.value.items()] if databases.value else [],
                            value=[d[0] for d in databases.value.items()][0] if databases.value else None,
                            interactive=True,
                            label="选择数据库",
                            elem_classes="llm-selector",
                        )
                        select_table = gr.Dropdown(
                            label="选择表",
                            interactive=True,
                            multiselect=True,
                            elem_classes="llm-selector",
                        )
                        is_dbqa = gr.Checkbox(
                            label="使用数据库问答",
                            value=False,
                            interactive=True,
                        )

                    with gr.Tab(label="参数"):
                        top_p = gr.Slider(
                            minimum=-0,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            interactive=True,
                            label="Top-p",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1,
                            step=0.1,
                            interactive=True,
                            label="Temperature",
                        )
                        max_tokens = gr.Slider(
                            minimum=0,
                            maximum=512,
                            value=512,
                            step=8,
                            interactive=True,
                            label="Max Generation Tokens",
                        )
                        memory_k = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=5,
                            step=1,
                            interactive=True,
                            label="Max Memory Window Size",
                        )
                        chunk_size = gr.Slider(
                            minimum=100,
                            maximum=1000,
                            value=200,
                            step=100,
                            interactive=True,
                            label="Chunk Size",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=10,
                            interactive=True,
                            label="Chunk Overlap",
                        )

    with gr.Tab(label="Query Result"), gr.Column():
        sql_res = gr.Dataframe(max_rows=10)

    gr.Markdown(description)

    add_model.click(
        add_llm,
        inputs=[model_name, api_base, models],
        outputs=[model_name, api_base, models, select_model],
    )

    add_database.click(
        add_db,
        inputs=[db_user, db_password, db_host, db_port, db_name, databases],
        outputs=[db_user, db_password, db_host, db_port, db_name, databases, select_database],
    )

    select_database.change(
        get_table_names,
        inputs=[select_database, databases],
        outputs=select_table,
    )

    file.upload(
        upload_file,
        inputs=file,
        outputs=select_file,
    )

    add_vs.click(
        add_vector_store,
        inputs=[select_file, select_model, models, chunk_size, chunk_overlap],
        outputs=status_display,
    )

    predict_args = dict(
        fn=predict,
        inputs=[
            select_model,
            models,
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_tokens,
            memory_k,
            is_kgqa,
            single_turn,
            is_dbqa,
            select_database,
            select_table,
            databases,
        ],
        outputs=[chatbot, history, status_display, sql_res],
        show_progress=True,
    )
    retry_args = dict(
        fn=retry,
        inputs=[
            select_model,
            models,
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_tokens,
            memory_k,
            is_kgqa,
            single_turn,
            is_dbqa,
            select_database,
            select_table,
            databases,
        ],
        outputs=[chatbot, history, status_display, sql_res],
        show_progress=True,
    )

    reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])

    cancelBtn.click(cancel_outputing, [], [status_display])
    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[user_input],
        outputs=[user_question, user_input, submitBtn, cancelBtn],
        show_progress=True,
    )

    user_input.submit(**transfer_input_args).then(**predict_args)

    submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    retryBtn.click(**retry_args)

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )

demo.title = "OpenLLM Chatbot 🚀 "

if __name__ == "__main__":
    reload_javascript()
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0")
